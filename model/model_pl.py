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


class CLIPReIDModuleStage1(pl.LightningModule):
    def __init__(self, cfg, num_classes, camera_num, view_num, num_query):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.num_query = num_query
        
        self.model = make_model(
            cfg, num_class=self.num_classes, camera_num=self.camera_num, view_num=self.view_num
        )
        for n, p in self.model.named_parameters():
            print(n, p.device, p.requires_grad)
        
        self.loss_meter = AverageMeter()
        self.text_features = None
        
        self.automatic_optimization = False
    
    def configure_optimizers(self):
        optimizer_1stage = make_optimizer_1stage(self.cfg, self.model)
        scheduler_1stage = create_scheduler(
            optimizer_1stage,
            num_epochs=self.cfg.training.solver.stage1.max_epochs,
            lr_min=self.cfg.training.solver.stage1.lr_min,
            warmup_lr_init=self.cfg.training.solver.stage1.warmup_lr_init,
            warmup_t=self.cfg.training.solver.stage1.warmup_epochs,
            noise_range=None,
        )
        scheduler_1stage = {
            'scheduler': scheduler_1stage,
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer_1stage], [scheduler_1stage]
    
    def on_train_start(self):
        self.xent = SupConLoss(device=self.device)
        
        self.extract_image_features()
    
    def extract_image_features(self):
        image_features = []
        labels = []        
        dataloader = self.trainer.datamodule.train_dataloader()  
        with torch.no_grad():
            for _, (img, vid, _, _) in enumerate(dataloader):
                img = img.to(self.device)
                target = vid
                with amp.autocast(self.device.type, enabled=True):
                    image_feature = self.model(img, target.to(self.device), get_image=True)
                    for i, img_feat in zip(target, image_feature):
                        labels.append(i.cpu())
                        image_features.append(img_feat.cpu())
            
            self.labels_list = torch.stack(labels, dim=0)
            self.image_features_list = torch.stack(image_features, dim=0)
            
            self.batch_size = self.cfg.training.solver.stage1.ims_per_batch
            self.num_image = self.labels_list.shape[0]
            self.i_ter = self.num_image // self.batch_size
            print("Iter first stage: ", self.i_ter)
        
        self.img_shape = img.shape

        self.model.train()
        del labels, image_features
        torch.cuda.empty_cache()
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        if batch_idx != self.i_ter:
            b_list = self.iter_list[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        else:
            b_list = self.iter_list[batch_idx * self.batch_size : self.num_image]
        
        target = self.labels_list[b_list].to(self.device)
        image_features = self.image_features_list[b_list].to(self.device)
        with amp.autocast(self.device.type, enabled=True):
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
    
    def on_train_epoch_start(self):
        self.loss_meter.reset()
        self.model.train()
        self.iter_list = torch.randperm(self.num_image)
        # self.trainer.fit_loop.epoch_loop.max_steps = self.i_ter

    def on_train_epoch_end(self):
        scheduler_stage1 = self.lr_schedulers()
        self.log(
            "base_lr_stage1",
            scheduler_stage1._get_lr(self.current_epoch)[0],
            on_epoch=True,
            logger=True,
        )
        scheduler_stage1.step(self.current_epoch + 1)
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def on_predict_epoch_start(self):
        self.evaluator = R1_mAP_eval(self.num_query, max_rank=50, feat_norm=self.cfg.testing.feat_norm)
        self.evaluator.reset()
        self.model.eval()
    
    def predict_step(self, batch, batch_idx):
        img, pid, camid, camids, target_view, _ = batch

        img = img
        
        if self.cfg.model.sie_camera:
            camids = camids
        else:
            camids = None
            
        if self.cfg.model.sie_view:
            target_view = target_view
        else:
            target_view = None
            
        feat = self.model(img, cam_label=camids, view_label=target_view)
        self.evaluator.update((feat, pid, camid))
    
    def on_predict_epoch_end(self):
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        
        self.log(
            "test_mAP",
            mAP,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "test_rank1",
            cmc[0],
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "test_rank5",
            cmc[4],
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "test_rank10",
            cmc[9],
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        


class CLIPReIDModuleStage2(pl.LightningModule):
    def __init__(self, cfg, model, num_classes, num_query):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_query = num_query
        
        self.model = model
        
        self.loss_fn, self.center_criterion = make_loss(cfg, num_classes=self.num_classes)
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()

        self.evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.testing.feat_norm)
        self.evaluator.reset()
        self.text_features = None
        
        self.automatic_optimization = False
    
    def configure_optimizers(self):
        optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(
            self.cfg, self.model, self.center_criterion
        )
        scheduler_2stage = WarmupMultiStepLR(
            optimizer_2stage,
            self.cfg.training.solver.stage2.steps,
            self.cfg.training.solver.stage2.gamma,
            self.cfg.training.solver.stage2.warmup_factor,
            self.cfg.training.solver.stage2.warmup_iters,
            self.cfg.training.solver.stage2.warmup_method,
        )
        scheduler_2stage = {
            'scheduler': scheduler_2stage,
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer_2stage, optimizer_center_2stage], [scheduler_2stage]
    
    def on_train_start(self):
        for n, p in self.model.named_parameters():
            print(n, p.device, p.requires_grad)
        
        self.extract_text_features()
    
    def extract_text_features(self):
        text_features = []
        batch = self.cfg.training.solver.stage2.ims_per_batch
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
                with amp.autocast(self.device.type, enabled=True):
                    text_feature = self.model(label=l_list.to(self.device), get_text=True)
                text_features.append(text_feature.cpu())
            
            self.text_features = torch.cat(text_features, 0).to(self.device)
        torch.cuda.empty_cache()
    
    def training_step(self, batch, batch_idx):
        optimizer, optimizer_center = self.optimizers()
        
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        
        img, vid, target_cam, target_view = batch
        
        target = vid.to(self.device)
        img = img.to(self.device)
        
        if self.cfg.model.sie_camera:
            target_cam = target_cam.to(self.device)
        else:
            target_cam = None
            
        if self.cfg.model.sie_view:
            target_view = target_view.to(self.device)
        else:
            target_view = None
        
        with amp.autocast(self.device.type, enabled=True):
            score, feat, image_features = self.model(
                x=img, label=target, cam_label=target_cam, view_label=target_view
            )
            logits = image_features @ self.text_features.t()
            loss = self.loss_fn(score, feat, target, target_cam, logits)
        
        self.manual_backward(loss)
        optimizer.step()
        
        if "center" in self.cfg.model.metric_loss_type:
            for param in self.center_criterion.parameters():
                param.grad.data *= (1.0 / self.cfg.training.solver.stage2.center_loss_weight)
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
        self.loss_meter.reset()
        self.acc_meter.reset()
        self.model.train()

    def on_train_epoch_end(self):
        scheduler_stage2 = self.lr_schedulers()
        self.log(
            "base_lr_stage2",
            scheduler_stage2.get_lr()[0],
            on_epoch=True,
            logger=True,
        )
        scheduler_stage2.step()
    
    def validation_step(self, batch, batch_idx):
        img, pid, camid, camids, target_view, _ = batch

        img = img
        
        if self.cfg.model.sie_camera:
            camids = camids
        else:
            camids = None
            
        if self.cfg.model.sie_view:
            target_view = target_view
        else:
            target_view = None
            
        feat = self.model(img, cam_label=camids, view_label=target_view)
        self.evaluator.update((feat, pid, camid))
    
    def on_validation_epoch_end(self):
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
        torch.cuda.empty_cache()
