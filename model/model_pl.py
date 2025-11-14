import io
import random
import time

import pytorch_lightning as pl
import torch
from torch import amp

from loss.make_loss import make_loss
from loss.supcontrast import SupConLoss
from loss.clip_contrastive_loss import CLIPContrastiveLoss
from solver.lr_scheduler import WarmupMultiStepLR
from solver.make_optimizer_prompt import make_optimizer_stage0, make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval

from .make_model_clipreid import make_model
from .train_logging import log_similar_texts_and_images


class CLIPReIDModuleStage0(pl.LightningModule):
    """
    Stage0: CLIP pretraining with image-text pairs from Market1501 captions.
    Unfreezes and jointly trains image_encoder, text_encoder, and token_embedding.
    """
    def __init__(self, cfg, num_classes, camera_num, view_num, num_query):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        
        self.model = make_model(
            cfg,
            num_class=self.num_classes,
            camera_num=self.camera_num,
            view_num=self.view_num,
        )

        self.clip_loss_fn = CLIPContrastiveLoss(
            temperature=cfg.training.solver.stage0.clip_temperature
        )

        self.clip_loss_meter = AverageMeter()
        self.i2t_acc_meter = AverageMeter()
        self.t2i_acc_meter = AverageMeter()
        
        self.automatic_optimization = False
        self.batch_size = self.cfg.training.solver.stage0.ims_per_batch
        
    def configure_optimizers(self):
        optimizer_stage0 = make_optimizer_stage0(self.cfg, self.model)
        
        scheduler_stage0 = WarmupMultiStepLR(
            optimizer_stage0,
            self.cfg.training.solver.stage0.steps,
            self.cfg.training.solver.stage0.gamma,
            self.cfg.training.solver.stage0.warmup_factor,
            self.cfg.training.solver.stage0.warmup_epochs,
            self.cfg.training.solver.stage0.warmup_method,
        )
        scheduler_stage0 = {
            "scheduler": scheduler_stage0,
            "interval": "epoch",
            "frequency": 1,
        }
        
        return [optimizer_stage0], [scheduler_stage0]
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        img, _, _, _, captions = batch
        img = img.to(self.device)
        
        with amp.autocast(self.device.type, enabled=True):
            features = self.model(x=img, captions=captions, is_stage0=True)
            image_features = features['image_features']
            text_features = features['text_features']

            clip_loss, _, _ = self.clip_loss_fn(image_features, text_features)
            i2t_acc, t2i_acc = self.clip_loss_fn.compute_accuracy(image_features, text_features)
        
        self.manual_backward(clip_loss)
        optimizer.step()

        self.clip_loss_meter.update(clip_loss.item(), self.batch_size)
        self.i2t_acc_meter.update(i2t_acc, self.batch_size)
        self.t2i_acc_meter.update(t2i_acc, self.batch_size)
        
        self.log(
            "train_loss_stage0",
            self.clip_loss_meter.avg,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train_acc_i2t_stage0",
            self.i2t_acc_meter.avg,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train_acc_t2i_stage0",
            self.t2i_acc_meter.avg,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        
        return clip_loss
    
    def validation_step(self, batch, batch_idx):
        img, _, _, _, captions = batch
        img = img.to(self.device)

        features = self.model(x=img, captions=captions, is_stage0=True)
        image_features = features['image_features']
        text_features = features['text_features']

        clip_loss, i2t_loss, t2i_loss = self.clip_loss_fn(image_features, text_features)
        i2t_acc, t2i_acc = self.clip_loss_fn.compute_accuracy(image_features, text_features)
        
        self.log(
            "val_loss_stage0",
            clip_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "val_acc_i2t_stage0",
            i2t_acc,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "val_acc_t2i_stage0",
            t2i_acc,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        return {"val_loss": clip_loss, "val_i2t_acc": i2t_acc, "val_t2i_acc": t2i_acc}
        
    def on_train_epoch_start(self):
        self.clip_loss_meter.reset() 
        self.i2t_acc_meter.reset()
        self.t2i_acc_meter.reset()
        
        self.model.train()

    def on_train_epoch_end(self):
        scheduler_stage0 = self.lr_schedulers()
        self.log(
            "base_lr_stage0",
            scheduler_stage0.get_lr()[0],
            on_epoch=True,
            logger=True,
        )
        scheduler_stage0.step(self.current_epoch + 1)

    def on_validation_epoch_end(self):
        pass


class CLIPReIDModuleStage1(pl.LightningModule):
    def __init__(self, cfg, num_classes, camera_num, view_num, num_query, model=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.num_query = num_query

        if model is None:
            self.model = make_model(
                cfg,
                num_class=self.num_classes,
                camera_num=self.camera_num,
                view_num=self.view_num,
            )
        else:
            self.model = model
        # for n, p in self.model.named_parameters():
        #     print(n, p.device, p.requires_grad)

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
            "scheduler": scheduler_1stage,
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer_1stage], [scheduler_1stage]

    def on_train_start(self):
        self.xent = SupConLoss(device=self.device)

        self.extract_image_features()

    def extract_image_features(self):
        image_features = []
        labels = []
        captions_list = []
        dataloader = self.trainer.train_dataloader
        with torch.no_grad():
            captions_in_train = 0
            for _, batch in enumerate(dataloader):
                if len(batch) == 5:  # ImageDatasetWithCaptions
                    img, vid, _, _, captions = batch
                    # print("Captions from extract image features: ", captions)
                else:
                    img, vid = batch[0], batch[1]
                    captions = [None] * len(vid)  # No captions available

                img = img.to(self.device)
                target = vid
                with amp.autocast(self.device.type, enabled=True):
                    image_feature = self.model(
                        img, target.to(self.device), get_image=True
                    )
                    for i, (target_id, img_feat) in enumerate(
                        zip(target, image_feature)
                    ):
                        labels.append(target_id.cpu())
                        image_features.append(img_feat.cpu())
                        if i < len(captions) and captions[i] and captions[i].strip():
                            captions_in_train += 1
                        captions_list.append(captions[i])
                # print("Captions in batch: ", captions_in_batch)

            self.labels_list = torch.stack(labels, dim=0)
            self.image_features_list = torch.stack(image_features, dim=0)
            self.captions_list = (
                captions_list
                if len([caption for caption in captions_list if caption is not None])
                > 0
                else None
            )

            self.batch_size = self.cfg.training.solver.stage1.ims_per_batch
            self.num_image = self.labels_list.shape[0]
            self.i_ter = self.num_image // self.batch_size
            print("Iter first stage: ", self.i_ter)
            print("Captions in train Stage 1: ", captions_in_train)
            print("Number of images in train Stage 1: ", self.num_image)

        self.model.train()
        del labels, image_features
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        if batch_idx != self.i_ter:
            b_list = self.iter_list[
                batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
            ]
        else:
            b_list = self.iter_list[batch_idx * self.batch_size : self.num_image]

        target = self.labels_list[b_list].to(self.device)
        image_features = self.image_features_list[b_list].to(self.device)

        # if self.captions_list is not None:
        #     batch_captions = [self.captions_list[i] for i in b_list.cpu().numpy()]
        # else:
        batch_captions = None

        with amp.autocast(self.device.type, enabled=True):
            text_features = self.model(
                label=target, get_text=True, captions=batch_captions
            )

        loss_i2t = self.xent(image_features, text_features, target, target)
        loss_t2i = self.xent(text_features, image_features, target, target)

        loss = loss_i2t + loss_t2i

        self.manual_backward(loss)
        optimizer.step()

        loss_value = loss.item()
        if not torch.isnan(loss) and not torch.isinf(loss):
            self.loss_meter.update(loss_value, self.batch_size)
        else:
            print(f"⚠️ Warning: Invalid loss value detected: {loss_value}")

        if self.loss_meter.avg is None:
            print("loss_meter.avg is None")

        self.log(
            "train_loss_stage1",
            self.loss_meter.avg,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
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
        self.evaluator = R1_mAP_eval(
            self.num_query, max_rank=50, feat_norm=self.cfg.testing.feat_norm
        )
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


class CLIPReIDModuleStage2(pl.LightningModule):
    def __init__(self, cfg, model, num_classes, num_query):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_query = num_query

        self.model = model

        self.loss_fn, self.center_criterion = make_loss(
            cfg, num_classes=self.num_classes
        )
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()

        self.evaluator = R1_mAP_eval(
            num_query, max_rank=50, feat_norm=cfg.testing.feat_norm
        )
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
            "scheduler": scheduler_2stage,
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer_2stage, optimizer_center_2stage], [scheduler_2stage]

    def on_train_start(self):
        # for n, p in self.model.named_parameters():
        #     print(n, p.device, p.requires_grad)

        # self.use_graph_sampling = (
        #     self.trainer.datamodule.use_graph_sampling
        #     and self.trainer.datamodule._model_for_sampling is not None
        # )

        self.extract_text_features()

    def extract_text_features(self):
        text_features = []
        batch = self.cfg.training.solver.stage2.ims_per_batch
        i_ter = self.num_classes // batch
        left = self.num_classes - batch * (self.num_classes // batch)
        if left != 0:
            i_ter = i_ter + 1

        class_captions = self._get_representative_captions_for_classes()
        # class_captions = {}

        with torch.no_grad():
            cnt_captions_in_train = 0
            for i in range(i_ter):
                if i + 1 != i_ter:
                    l_list = torch.arange(i * batch, (i + 1) * batch)
                else:
                    l_list = torch.arange(i * batch, self.num_classes)

                batch_captions = []
                for class_id in l_list:
                    caption = class_captions.get(class_id.item(), None)

                    if caption is not None:
                        use_caption = random.random() < 0.7
                        if use_caption:
                            batch_captions.append(caption)
                            cnt_captions_in_train += 1
                        else:
                            batch_captions.append(None)
                    else:
                        batch_captions.append(None)
                batch_captions = (
                    batch_captions
                    if len(
                        [caption for caption in batch_captions if caption is not None]
                    )
                    > 0
                    else None
                )

                with amp.autocast(self.device.type, enabled=True):
                    text_feature = self.model(
                        label=l_list.to(self.device),
                        get_text=True,
                        captions=batch_captions,
                    )
                text_features.append(text_feature.cpu())

            self.text_features = torch.cat(text_features, 0).to(self.device)
            self.cnt_captions_in_train = cnt_captions_in_train
            print(
                "Captions found for classes in train Stage 2: ", cnt_captions_in_train
            )
            print("Number of classes in train Stage 2: ", self.num_classes)

        torch.cuda.empty_cache()

    def _get_representative_captions_for_classes(self):
        class_to_captions = {}
        total_cnt_classes_with_captions = 0
        dataloader = self.trainer.train_dataloader
        for batch in dataloader:
            if len(batch) == 5:
                _, vids, _, _, captions = batch
            else:
                break  # Dataloader without captions

            for vid, caption in zip(vids, captions):
                if caption is None:
                    continue
                class_id = vid.item()
                if class_id not in class_to_captions:
                    class_to_captions[class_id] = []
                    total_cnt_classes_with_captions += 1
                class_to_captions[class_id].append(caption)

        print(f"Stage 2: Found {total_cnt_classes_with_captions} classes with captions")

        class_unique_captions = {}
        for class_id, captions in class_to_captions.items():
            class_unique_captions[class_id] = random.choice(captions)

        return class_unique_captions

    def training_step(self, batch, batch_idx):
        optimizer, optimizer_center = self.optimizers()

        optimizer.zero_grad()
        optimizer_center.zero_grad()

        img, vid, target_cam, target_view = batch[:4]

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
                param.grad.data *= (
                    1.0 / self.cfg.training.solver.stage2.center_loss_weight
                )
            optimizer_center.step()

        if (
            self.cfg.training.solver.text_features_viz
            and self.global_step % self.cfg.training.solver.text_features_viz_frequency
            == 0
        ):
            log_similar_texts_and_images(
                self,
                img,
                target,
                self.current_epoch,
                self.global_step,
                n_similar=self.cfg.training.solver.viz_n_similar
            )

        acc = (logits.max(1)[1] == target).float().mean()
        self.loss_meter.update(loss.item(), len(vid))
        self.acc_meter.update(acc, len(vid))

        self.log(
            "train_loss_stage2",
            self.loss_meter.avg,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train_acc_stage2",
            self.acc_meter.avg,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def on_train_epoch_start(self):
        self.loss_meter.reset()
        self.acc_meter.reset()
        self.model.train()

        if self.current_epoch > 1 and self.cnt_captions_in_train > 0 and self.current_epoch % 10 == 0:
            print(
                f"🔄 Stage 2 - Epoch {self.current_epoch}: Re-selecting random captions..."
            )
            self.extract_text_features()

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

    def on_validation_epoch_start(self):
        self.evaluator.reset()
        self.model.eval()

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
