import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader
from pathlib import Path

# from .sampler_ddp import RandomIdentitySampler_DDP
# from .vehicleid import VehicleID
# from .veri import VeRi
from configs.constants import DEVICE

from .bases import ImageDataset, ImageDatasetWithCaptions
from .graph_sampler import TextGraphSampler

# from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501

# from .msmt17 import MSMT17
# from .occ_duke import OCC_DukeMTMCreID
from .sampler import RandomIdentitySampler

factory = {
    "market1501": Market1501,
    # "dukemtmc": DukeMTMCreID,
    # "msmt17": MSMT17,
    # "occ_duke": OCC_DukeMTMCreID,
    # "veri": VeRi,
    # "VehicleID": VehicleID,
}


def train_collate_fn(batch):
    if len(batch[0]) == 6:
        imgs, pids, camids, viewids, img_filenames, captions = zip(*batch)
    else:
        imgs, pids, camids, viewids, img_filenames = zip(*batch)
        captions = False
    
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    
    if captions:
        return (
            torch.stack(imgs, dim=0),
            pids,
            camids,
            viewids,
            captions,
        )
    
    return (
        torch.stack(imgs, dim=0),
        pids,
        camids,
        viewids,
    )


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


class CLIPReIDDataModuleStage1(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg.dataset.names
        self.root_dir = cfg.dataset.root_dir
        self.data_dir = cfg.dataset.data_dir
        self.dataset_dir = cfg.dataset.dataset_dir
        self.batch_size_stage1 = cfg.training.solver.stage1.ims_per_batch
        self.test_batch_size = cfg.testing.ims_per_batch
        self.num_workers = cfg.training.dataloader.num_workers
        self.sampler = cfg.training.dataloader.sampler
        self.num_instance = cfg.training.dataloader.num_instance
        
        # Graph sampling parameters
        self.use_graph_sampling = cfg.training.dataloader.use_graph_sampling
        self.graph_sampling_verbose = getattr(cfg.training.dataloader, 'graph_sampling_verbose', False)
        self._model_for_sampling = None
        
        if "captions_json" not in cfg.dataset:
            self.captions_json_path = None
        else:
            self.captions_json_path = Path(cfg.dataset.root_dir) / cfg.dataset.data_dir / cfg.dataset.captions_json

        self.train_transforms = T.Compose(
            [
                T.Resize(cfg.preprocessing.size_train, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.preprocessing.prob),
                T.Pad(cfg.preprocessing.padding),
                T.RandomCrop(cfg.preprocessing.size_train),
                T.ToTensor(),
                T.Normalize(
                    mean=cfg.preprocessing.pixel_mean, std=cfg.preprocessing.pixel_std
                ),
                RandomErasing(
                    probability=cfg.preprocessing.re_prob,
                    mode=cfg.preprocessing.re_mode,
                    max_count=cfg.preprocessing.re_max_count,
                    device=DEVICE,
                ),
            ]
        )

        self.val_transforms = T.Compose(
            [
                T.Resize(cfg.preprocessing.size_test),
                T.ToTensor(),
                T.Normalize(
                    mean=cfg.preprocessing.pixel_mean, std=cfg.preprocessing.pixel_std
                ),
            ]
        )

    def setup(self, stage=None):
        self.dataset = factory[self.dataset_name](
            root=self.root_dir, data_dir=self.data_dir, datast_dir=self.dataset_dir
        )
        self.num_classes = self.dataset.num_train_pids
        self.cam_num = self.dataset.num_train_cams
        self.view_num = self.dataset.num_train_vids
        self.num_query = len(self.dataset.query)

        if self.captions_json_path:
            self.train_set_normal = ImageDatasetWithCaptions(
                self.dataset.train, 
                self.val_transforms, 
                self.captions_json_path
            )
        else:
            self.train_set_normal = ImageDataset(self.dataset.train, self.val_transforms)
        
        self.val_set = ImageDataset(
            self.dataset.query + self.dataset.gallery, self.val_transforms
        )

    def train_dataloader(self):
        if self.use_graph_sampling and self._model_for_sampling is not None:       
            sampler = TextGraphSampler(
                data_source=self.dataset.train,
                model=self._model_for_sampling,
                captions_map=getattr(self.train_set_normal, 'captions', {}),
                batch_size=self.batch_size_stage1,
                num_instance=self.num_instance,
                verbose=self.graph_sampling_verbose
            )
            
            return DataLoader(
                self.train_set_normal,
                batch_size=self.batch_size_stage1,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=train_collate_fn,
                persistent_workers=True,
            )

        else:
            return DataLoader(
                self.train_set_normal,
                batch_size=self.batch_size_stage1,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=train_collate_fn,
                persistent_workers=True,
            )
    
    def set_model_for_graph_sampling(self, model):
        self._model_for_sampling = model

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=val_collate_fn,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return self.val_dataloader()

    def get_dataset_info(self):
        return {
            "num_query": self.num_query,
            "num_classes": self.num_classes,
            "camera_num": self.cam_num,
            "view_num": self.view_num,
        }


class CLIPReIDDataModuleStage2(pl.LightningDataModule):
    def __init__(self, cfg, downoladed_dataset):
        super().__init__()
        self.cfg = cfg
        self.dataset = downoladed_dataset
        self.batch_size_stage2 = cfg.training.solver.stage2.ims_per_batch
        self.test_batch_size = cfg.testing.ims_per_batch
        self.num_workers = cfg.training.dataloader.num_workers
        self.sampler = cfg.training.dataloader.sampler
        self.num_instance = cfg.training.dataloader.num_instance

        # Graph sampling parameters
        self.use_graph_sampling = False
        self.graph_sampling_verbose = False
        self._model_for_sampling = None
        
        if "captions_json" not in cfg.dataset:
            self.captions_json_path = None
        else:
            self.captions_json_path = Path(cfg.dataset.root_dir) / cfg.dataset.data_dir / cfg.dataset.captions_json

        self.train_transforms = T.Compose(
            [
                T.Resize(cfg.preprocessing.size_train, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.preprocessing.prob),
                T.Pad(cfg.preprocessing.padding),
                T.RandomCrop(cfg.preprocessing.size_train),
                T.ToTensor(),
                T.Normalize(
                    mean=cfg.preprocessing.pixel_mean, std=cfg.preprocessing.pixel_std
                ),
                RandomErasing(
                    probability=cfg.preprocessing.re_prob,
                    mode=cfg.preprocessing.re_mode,
                    max_count=cfg.preprocessing.re_max_count,
                    device=DEVICE,
                ),
            ]
        )

        self.val_transforms = T.Compose(
            [
                T.Resize(cfg.preprocessing.size_test),
                T.ToTensor(),
                T.Normalize(
                    mean=cfg.preprocessing.pixel_mean, std=cfg.preprocessing.pixel_std
                ),
            ]
        )

    def setup(self, stage=None):
        self.num_classes = self.dataset.num_train_pids
        self.cam_num = self.dataset.num_train_cams
        self.view_num = self.dataset.num_train_vids
        self.num_query = len(self.dataset.query)

        if self.captions_json_path:
            self.train_set = ImageDatasetWithCaptions(
                self.dataset.train, 
                self.train_transforms, 
                self.captions_json_path
            )
        else:
            self.train_set = ImageDataset(self.dataset.train, self.train_transforms)

        self.val_set = ImageDataset(
            self.dataset.query + self.dataset.gallery, self.val_transforms
        )

    def train_dataloader(self):
        if "triplet" in self.sampler:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size_stage2,
                sampler=RandomIdentitySampler(
                    self.dataset.train,
                    self.batch_size_stage2,
                    self.num_instance,
                ),
                num_workers=self.num_workers,
                collate_fn=train_collate_fn,
                persistent_workers=True,
            )
        elif self.sampler == "softmax":
            print("using softmax sampler")
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size_stage2,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=train_collate_fn,
            )
        else:
            print(
                "unsupported sampler! expected softmax or triplet but got {}".format(
                    self.sampler
                )
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=val_collate_fn,
            persistent_workers=True,
        )

    def get_dataset_info(self):
        return {
            "num_query": self.num_query,
            "num_classes": self.num_classes,
            "camera_num": self.cam_num,
            "view_num": self.view_num,
        }
