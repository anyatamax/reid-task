import torch
import torch.distributed as dist
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .bases import ImageDataset
# from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
# from .msmt17 import MSMT17
# from .occ_duke import OCC_DukeMTMCreID
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
# from .vehicleid import VehicleID
# from .veri import VeRi

factory = {
    "market1501": Market1501,
    # "dukemtmc": DukeMTMCreID,
    # "msmt17": MSMT17,
    # "occ_duke": OCC_DukeMTMCreID,
    # "veri": VeRi,
    # "VehicleID": VehicleID,
}

def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
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


class CLIPReIDDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg.DATASETS.NAMES
        self.root_dir = cfg.DATASETS.ROOT_DIR
        self.batch_size_stage1 = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        self.batch_size_stage2 = cfg.SOLVER.STAGE2.IMS_PER_BATCH
        self.test_batch_size = cfg.TEST.IMS_PER_BATCH
        self.num_workers = cfg.DATALOADER.NUM_WORKERS
        self.sampler = cfg.DATALOADER.SAMPLER
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        
        self.train_transforms = T.Compose(
            [
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(
                    probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"
                ),
            ]
        )

        self.val_transforms = T.Compose(
            [
                T.Resize(cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ]
        )
        
    
    def setup(self, stage = None):
        self.dataset = factory[self.cfg.DATASETS.NAMES](root=self.cfg.DATASETS.ROOT_DIR)
        self.num_classes = self.dataset.num_train_pids
        self.cam_num = self.dataset.num_train_cams
        self.view_num = self.dataset.num_train_vids
        self.num_query = len(self.dataset.query)

        self.train_set = ImageDataset(self.dataset.train, self.train_transforms)
        self.train_set_normal = ImageDataset(self.dataset.train, self.val_transforms)
        self.val_set = ImageDataset(self.dataset.query + self.dataset.gallery, self.val_transforms)


    def stage1_dataloader(self):
        return DataLoader(
            self.train_set_normal,
            batch_size=self.batch_size_stage1,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=train_collate_fn,
        )


    def train_dataloader(self):
        if "triplet" in self.sampler:
            if self.cfg.MODEL.DIST_TRAIN:
                print("DIST_TRAIN START")
                mini_batch_size = self.batch_size_stage2 // dist.get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    self.dataset.train,
                    self.batch_size_stage2,
                    self.num_instance,
                )
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, 
                    mini_batch_size,
                    True
                )
                return DataLoader(
                    self.train_set,
                    num_workers=self.num_workers,
                    batch_sampler=batch_sampler,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                )
            else:
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
        )
    
    def test_dataloader(self):
        return self.val_dataloader()

    def get_dataset_info(self):
        return {
            "num_query": self.num_query,
            "num_classes": self.num_classes,
            "camera_num": self.cam_num,
            "view_num": self.view_num
        }
