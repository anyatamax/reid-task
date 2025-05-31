import argparse
import os
import random
import datetime

import numpy as np
import torch
import torch.multiprocessing as mp

from datasets.data import CLIPReIDDataModule
from model.model_pl import CLIPReIDModule
from configs.constants import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, ModelSummary, Timer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.training.solver.seed)

    output_dir = cfg.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model in the path : {}".format(output_dir))

    data_module = CLIPReIDDataModule(cfg)
    data_module.setup()

    dataset_info = data_module.get_dataset_info()

    model = CLIPReIDModule(
        cfg=cfg,
        num_classes=dataset_info["num_classes"],
        camera_num=dataset_info["camera_num"],
        view_num=dataset_info["view_num"],
        num_query=dataset_info["num_query"]
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename=f"{cfg.model.name}" + "-{epoch:02d}-{val_rank1:.4f}",
            monitor="val_rank1",
            mode="max",
            save_top_k=SAVE_TOP_K,
            save_last=SAVE_LAST,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val_rank1",
            patience=EARLY_STOPPING_PATIENCE,
            mode=EARLY_STOPPING_MODE,
            verbose=True,
        ),
        Timer()
    ]
    
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=cfg.logging.tesorboard_dir,
        default_hp_metric=False,
    )
    
    if DIST_TRAIN:
        strategy = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
            timeout=datetime.timedelta(seconds=1800),
            gradient_as_bucket_view=True,
        )
    else:
        strategy = "auto"
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.solver.stage1.max_epochs + cfg.training.solver.stage2.max_epochs,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        strategy=strategy,
        precision=PRECISION,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.training.solver.log_period,
        check_val_every_n_epoch=cfg.training.solver.eval_period,
        deterministic=DETERMINISTIC,
    )
    
    if DIST_TRAIN and mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()