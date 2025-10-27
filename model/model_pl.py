import io
import random
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import amp

from loss.make_loss import make_loss
from loss.supcontrast import SupConLoss
from solver.lr_scheduler import WarmupMultiStepLR
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval

from .make_model_clipreid import make_model


def plot_text_features_2d(
    text_features, target, batch_captions, current_epoch, global_step
):
    features_np = text_features.cpu().detach().numpy()
    labels_np = target.cpu().detach().numpy()

    valid_caption_indices = set()
    if batch_captions is not None:
        for i, caption in enumerate(batch_captions):
            if caption is not None and caption.strip():
                valid_caption_indices.add(i)

    features_tensor = torch.from_numpy(features_np)

    # Cosine similarity matrix
    normalized_features = F.normalize(features_tensor, p=2, dim=1)
    cosine_similarity_matrix = torch.matmul(
        normalized_features, normalized_features.t()
    )
    cosine_distance_matrix = 1.0 - cosine_similarity_matrix

    # Euclidean distance matrix
    euclidean_distance_matrix = torch.cdist(
        features_tensor.float(), features_tensor.float(), p=2
    )

    reducer = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(features_np) - 1)
    )
    features_2d = reducer.fit_transform(features_np)
    method_name = "t-SNE"

    plt.style.use("default")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Text Features Visualization - Epoch {current_epoch}, Step {global_step}",
        fontsize=16,
        fontweight="bold",
    )

    unique_labels = np.unique(labels_np)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    for i, (x, y) in enumerate(features_2d):
        label = labels_np[i]
        has_caption = i in valid_caption_indices

        if has_caption:
            # Точки с captions - большие, яркие, с черной обводкой
            ax1.scatter(
                x,
                y,
                c=[color_map[label]],
                s=120,
                alpha=0.9,
                edgecolors="black",
                linewidth=2,
                marker="o",
            )
            ax1.annotate(
                f"ID:{label}",
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=9,
                alpha=1.0,
                fontweight="bold",
            )
        else:
            # Точки без captions - меньше, прозрачнее, серая обводка
            ax1.scatter(
                x,
                y,
                c=[color_map[label]],
                s=60,
                alpha=0.4,
                edgecolors="gray",
                linewidth=0.5,
                marker="o",
            )
            ax1.annotate(
                f"ID:{label}",
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                alpha=0.6,
            )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=2,
            label=f"With Captions ({len(valid_caption_indices)})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            markeredgecolor="gray",
            markeredgewidth=0.5,
            label=f"Without Captions ({len(features_np) - len(valid_caption_indices)})",
            alpha=0.6,
        ),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    ax1.set_title(f"2D Projection ({method_name}) - ALL Features, Captions Highlighted")
    ax1.set_xlabel(f"{method_name} Component 1")
    ax1.set_ylabel(f"{method_name} Component 2")
    ax1.grid(True, alpha=0.3)

    # График 2: Heatmap косинусных расстояний
    im1 = ax2.imshow(cosine_distance_matrix.numpy(), cmap="viridis", aspect="auto")
    ax2.set_title("Cosine Distance Matrix")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Sample Index")
    plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)

    # График 3: Heatmap евклидовых расстояний
    im2 = ax3.imshow(euclidean_distance_matrix.numpy(), cmap="plasma", aspect="auto")
    ax3.set_title("Euclidean Distance Matrix")
    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("Sample Index")
    plt.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)

    # График 4: Гистограмма расстояний
    cosine_distances_flat = cosine_distance_matrix[
        torch.triu(torch.ones_like(cosine_distance_matrix, dtype=bool), diagonal=1)
    ]
    euclidean_distances_flat = euclidean_distance_matrix[
        torch.triu(torch.ones_like(euclidean_distance_matrix, dtype=bool), diagonal=1)
    ]

    ax4.hist(
        cosine_distances_flat.numpy(),
        bins=20,
        alpha=0.7,
        label="Cosine Distance",
        color="blue",
    )
    ax4.hist(
        euclidean_distances_flat.numpy(),
        bins=20,
        alpha=0.7,
        label="Euclidean Distance",
        color="red",
    )
    ax4.set_title("Distance Distribution")
    ax4.set_xlabel("Distance")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 4. Сохраняем в MLflow как изображение
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    img_buffer.seek(0)

    try:
        img = Image.open(img_buffer)
        mlflow.log_image(
            image=img,
            artifact_file=f"text_features_viz/epoch_{current_epoch}_step_{global_step}_2d_viz.png",
        )
    except:
        img_buffer.seek(0)
        mlflow.log_artifact(
            img_buffer,
            f"text_features_viz/epoch_{current_epoch}_step_{global_step}_2d_viz.png",
        )

    plt.close(fig)

    print(f"✅ 2D visualization saved to MLflow artifacts!")


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
            cfg,
            num_class=self.num_classes,
            camera_num=self.camera_num,
            view_num=self.view_num,
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
            "scheduler": scheduler_1stage,
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer_1stage], [scheduler_1stage]

    def on_train_start(self):
        self.xent = SupConLoss(device=self.device)

        self.use_graph_sampling = (
            self.trainer.datamodule.use_graph_sampling
            and self.trainer.datamodule._model_for_sampling is not None
        )

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

        self.img_shape = img.shape

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

        if self.captions_list is not None:
            batch_captions = [self.captions_list[i] for i in b_list.cpu().numpy()]
        else:
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

        if (
            self.cfg.training.solver.text_features_viz
            and self.global_step % self.cfg.training.solver.text_features_viz_frequency
            == 0
        ):
            plot_text_features_2d(
                text_features,
                target,
                batch_captions,
                self.current_epoch,
                self.global_step,
            )

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

        if self.use_graph_sampling:
            # and self.current_epoch > self.cfg.training.solver.stage1.max_epochs - self.cfg.training.solver.stage1.graph_sampling_epochs:
            self.trainer.datamodule._model_for_sampling = self.model

            # if self.current_epoch % 5 == 0:
            if self.current_epoch > 1:
                start_time = time.time()
                print(f"🔄 Rebuilding graph sampling indices...")
                self.trainer.train_dataloader.sampler.make_index()
                elapsed_time = time.time() - start_time
                print(f"✅ Graph sampling indices rebuilt in {elapsed_time:.2f}s")

                start_time = time.time()
                print(
                    f"🔄 Epoch {self.current_epoch}: Extracting fresh image features with new graph order..."
                )
                self.extract_image_features()
                elapsed_time = time.time() - start_time
                print(f"✅ Image features extracted in {elapsed_time:.2f}s")

            self.iter_list = torch.arange(self.num_image)
        else:
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

        class_captions = self._get_representative_captions_for_classes()

        with torch.no_grad():
            cnt_captions_in_train = 0
            for i in range(i_ter):
                if i + 1 != i_ter:
                    l_list = torch.arange(i * batch, (i + 1) * batch)
                else:
                    l_list = torch.arange(i * batch, self.num_classes)

                batch_captions = []
                for class_id in l_list:
                    batch_captions.append(class_captions.get(class_id.item(), None))
                    if class_captions.get(class_id.item(), None) is not None:
                        cnt_captions_in_train += 1
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
            if len(batch) == 6:
                _, vids, _, _, _, captions = batch
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

        if self.current_epoch > 1:
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
