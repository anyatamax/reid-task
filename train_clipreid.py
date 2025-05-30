import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from datasets.data import CLIPReIDDataModule
from model.model_pl import CLIPReIDModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, ModelSummary, Timer
from pytorch_lightning.loggers import TensorBoardLogger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file",
        default="configs/person/vit_clipreid.yml",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model in the path :{}".format(cfg.OUTPUT_DIR))

    if args.config_file != "":
        print("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            print("Config args: ", config_str)

    data_module = CLIPReIDDataModule(cfg)
    data_module.setup()

    dataset_info = data_module.get_dataset_info()

    model = CLIPReIDModule(
        cfg=cfg,
        num_classes=dataset_info["num_classes"],
        camera_num=dataset_info["cam_num"],
        view_num=dataset_info["view_num"],
        num_query=dataset_info["num_query"]
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename=f"{cfg.MODEL.NAME}" + "-{epoch:02d}-{val_rank1:.4f}",
            monitor="val_rank1",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val_rank1",
            patience=10,
            mode="max",
            verbose=True,
        ),
        ModelSummary(max_depth=5),
        Timer()
    ]
    
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="reid_logs",
        default_hp_metric=False,
    )
    
    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS + cfg.SOLVER.STAGE2.MAX_EPOCHS,
        accelerator="gpu",
        devices=[1,3,7],
        strategy="ddp",
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=cfg.SOLVER.STAGE2.EVAL_PERIOD,
        deterministic=True,
    )
    
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()