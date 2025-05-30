import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
import pytorch_lightning as pl

from .make_model_clipreid import make_model
from loss.make_loss import make_loss
from loss.supcontrast import SupConLoss
from utils.metrics import R1_mAP_eval
from solver.lr_scheduler import WarmupMultiStepLR
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from utils.meter import AverageMeter


class CLIPReIDModule(pl.LightningModule):
    def __init__(self, cfg, num_classes, camera_num, view_num, num_query):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.num_query = num_query
        
        self.model = make_model(
            cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num
        )
        
        self.loss_fn, self.center_criterion = make_loss(cfg, num_classes=num_classes)
        self.xent = SupConLoss(device=self.device)
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()

        self.evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

        self.stage = 1
        self.text_features = None
        
        self.automatic_optimization = False
    
    def configure_optimizers(self):
        optimizer_1stage = make_optimizer_1stage(self.cfg, self.model)
        scheduler_1stage = create_scheduler(
            optimizer_1stage,
            num_epochs=self.cfg.SOLVER.STAGE1.MAX_EPOCHS,
            lr_min=self.cfg.SOLVER.STAGE1.LR_MIN,
            warmup_lr_init=self.cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
            warmup_t=self.cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
            noise_range=None,
        )
        scheduler_1stage = {
            'scheduler': scheduler_1stage,
            'interval': 'epoch',
            'frequency': 1
        }

        optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(
            self.cfg, self.model, self.center_criterion
        )
        scheduler_2stage = WarmupMultiStepLR(
            optimizer_2stage,
            self.cfg.SOLVER.STAGE2.STEPS,
            self.cfg.SOLVER.STAGE2.GAMMA,
            self.cfg.SOLVER.STAGE2.WARMUP_FACTOR,
            self.cfg.SOLVER.STAGE2.WARMUP_ITERS,
            self.cfg.SOLVER.STAGE2.WARMUP_METHOD,
        )
        scheduler_2stage = {
            'scheduler': scheduler_2stage,
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer_1stage, optimizer_2stage, optimizer_center_2stage], [scheduler_1stage, scheduler_2stage]
    
    def on_train_start(self):
        if self.stage == 1:
            model_path = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.MODEL.MODEL_CHKP_NAME_STAGE1)
            if not os.path.exists(model_path):
                print("Not found checkpoint")
                self.extract_image_features()
            else:
                print("Loading from checkpoint {}".format(model_path))
                state_dict = torch.load(model_path, weights_only=True)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace("module.", "") if k.startswith("module.") else k
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict, strict=False)
                print("Successfully loaded checkpoint with 'module.' prefix handling")
                
                self.stage = 2
    
    def extract_image_features(self):
        self.model.eval()
        image_features = []
        labels = []

        dataloader = self.trainer.datamodule.stage1_dataloader()
        self.len_dataloader_stage1 = len(dataloader)        
        
        with torch.no_grad():
            for _, (img, vid, _, _) in enumerate(dataloader):
                img = img.to(self.device)
                target = vid
                with amp.autocast(self.device, enabled=True):
                    image_feature = self.model(img, target.to(self.device), get_image=True)
                    for i, img_feat in zip(target, image_feature):
                        labels.append(i.cpu())
                        image_features.append(img_feat.cpu())
            
            self.labels_list = torch.stack(labels, dim=0)
            self.image_features_list = torch.stack(image_features, dim=0)
            
            self.batch_size = self.cfg.SOLVER.STAGE1.IMS_PER_BATCH
            self.num_image = self.labels_list.shape[0]
            self.i_ter = self.num_image // self.batch_size
        
        self.img_shape = img.shape

        self.model.train()
        del labels, image_features
    
    def prepare_stage2(self):
        self.model.eval()
        text_features = []
        
        batch = self.cfg.SOLVER.STAGE2.IMS_PER_BATCH
        i_ter = self.num_classes // batch
        left = self.num_classes - batch * (self.num_classes // batch)
        if left != 0:
            i_ter = i_ter + 1
        
        with torch.no_grad():
            for i in range(i_ter):
                if i + 1 != i_ter:
                    l_list = torch.arange(i * batch, (i + 1) * batch)
                else:
                    l_list = torch.arange(i * batch, self.num_classes)
                with amp.autocast(self.device, enabled=True):
                    text_feature = self.model(label=l_list.to(self.device), get_text=True)
                text_features.append(text_feature.cpu())
            
            self.text_features = torch.cat(text_features, 0).to(self.device)
        
        self.stage = 2
    
    def training_step(self, batch, batch_idx):
        if self.stage == 1:
            return self.training_step_stage1(batch, batch_idx)
        else:
            return self.training_step_stage2(batch, batch_idx)
    
    def training_step_stage1(self, batch, batch_idx):
        optimizer = self.optimizers()[0]
        optimizer.zero_grad()

        if batch_idx != self.i_ter:
            b_list = self.iter_list[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        else:
            b_list = self.iter_list[batch_idx * self.batch_size : self.num_image]
        
        target = self.labels_list[b_list].to(self.device)
        image_features = self.image_features_list[b_list].to(self.device)
        
        with amp.autocast(self.device, enabled=True):
            text_features = self.model(label=target, get_text=True)
        
        loss_i2t = self.xent(image_features, text_features, target, target)
        loss_t2i = self.xent(text_features, image_features, target, target)
        
        loss = loss_i2t + loss_t2i

        self.manual_backward(loss)
        optimizer.step()
        
        self.loss_meter.update(loss.item(), self.img_shape[0])

        self.log(
            "train_loss_stage1",
            self.loss_meter.avg,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        
        return loss
    
    def training_step_stage2(self, batch, batch_idx):
        optimizer = self.optimizers()[1]
        optimizer_center = self.optimizers()[2]
        
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        
        img, vid, target_cam, target_view = batch
        
        target = vid.to(self.device)
        img = img.to(self.device)
        
        if self.cfg.MODEL.SIE_CAMERA:
            target_cam = target_cam.to(self.device)
        else:
            target_cam = None
            
        if self.cfg.MODEL.SIE_VIEW:
            target_view = target_view.to(self.device)
        else:
            target_view = None
        
        with amp.autocast(self.device, enabled=True):
            score, feat, image_features = self.model(
                x=img, label=target, cam_label=target_cam, view_label=target_view
            )
            logits = image_features @ self.text_features.t()
            loss = self.loss_fn(score, feat, target, target_cam, logits)
        
        self.manual_backward(loss)
        optimizer.step()
        
        if "center" in self.cfg.MODEL.METRIC_LOSS_TYPE:
            for param in self.center_criterion.parameters():
                param.grad.data *= (1.0 / self.cfg.SOLVER.CENTER_LOSS_WEIGHT)
            optimizer_center.step()
        
        acc = (logits.max(1)[1] == target).float().mean()
        self.loss_meter.update(loss.item(), img.shape[0])
        self.acc_meter.update(acc, 1)
        
        self.log(
            "train_loss_stage2",
            self.loss_meter.avg,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_acc_stage2",
            self.acc_meter.avg,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        
        return loss
    
    def on_train_epoch_start(self):
        if self.stage == 1:
            self.loss_meter.reset()
            self.model.train()
            self.iter_list = torch.randperm(self.num_image)
        else:
            self.loss_meter.reset()
            self.acc_meter.reset()
            self.model.train()

    def on_train_epoch_end(self):
        if self.stage == 1 and self.current_epoch >= self.cfg.SOLVER.STAGE1.MAX_EPOCHS - 1:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    os.path.join(self.cfg.OUTPUT_DIR, self.cfg.MODEL.MODEL_CHKP_NAME_STAGE1)
                ),
            )
            self.prepare_stage2()
        elif self.stage == 1:
            scheduler_stage1 = self.lr_schedulers()[0]
            self.log(
                "base_lr_stage1",
                scheduler_stage1._get_lr(self.current_epoch)[0],
                on_epoch=True,
                logger=True,
            )
            scheduler_stage1.step(self.current_epoch + 1)
        else:
            scheduler_stage2 = self.lr_schedulers()[1]
            self.log(
                "base_lr_stage2",
                scheduler_stage2._get_lr(self.current_epoch)[0],
                on_epoch=True,
                logger=True,
            )
            scheduler_stage2.step()
    
    def validation_step(self, batch, batch_idx):
        if self.stage == 2:
            img, pid, camid, camids, target_view, _ = batch
            
            with torch.no_grad():
                img = img.to(self.device)
                
                if self.cfg.MODEL.SIE_CAMERA:
                    camids = camids.to(self.device)
                else:
                    camids = None
                    
                if self.cfg.MODEL.SIE_VIEW:
                    target_view = target_view.to(self.device)
                else:
                    target_view = None
                    
                feat = self.model(img, cam_label=camids, view_label=target_view)
                self.evaluator.update((feat, pid, camid))
    
    def on_validation_epoch_end(self):
        if self.stage == 2:
            cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
            
            self.log(
                "val_mAP",
                mAP,
                prog_bar=True,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "val_rank1",
                cmc[0],
                prog_bar=True,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "val_rank5",
                cmc[4],
                prog_bar=True,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "val_rank10",
                cmc[9],
                prog_bar=True,
                logger=True,
                on_epoch=True,
            )
            
            self.evaluator.reset()
    
    # def on_test_epoch_start(self):
    #     self.evaluator.reset()
    #     self.model.eval()
    
    # def test_step(self, batch, batch_idx):
    #     return self.validation_step(batch, batch_idx)
    
    # def on_test_epoch_end(self):
    #     cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        
    #     self.log(
    #         "test_mAP",
    #         mAP,
    #         prog_bar=True,
    #         logger=True,
    #         on_epoch=True,
    #     )
    #     self.log(
    #         "test_rank1",
    #         cmc[0],
    #         prog_bar=True,
    #         logger=True,
    #         on_epoch=True,
    #     )
    #     self.log(
    #         "test_rank5",
    #         cmc[4],
    #         prog_bar=True,
    #         logger=True,
    #         on_epoch=True,
    #     )
    #     self.log(
    #         "test_rank10",
    #         cmc[9],
    #         prog_bar=True,
    #         logger=True,
    #         on_epoch=True,
    #     )
        
    #     self.print(f"Test Results - Rank-1: {cmc[0]:.1%}")
    #     self.print(f"Test Results - Rank-5: {cmc[4]:.1%}")
    #     self.print(f"Test Results - mAP: {mAP:.1%}")

