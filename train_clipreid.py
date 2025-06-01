import argparse
import os
import random
import datetime

import numpy as np
import torch
import torch.multiprocessing as mp

from datasets.data import CLIPReIDDataModuleStage1, CLIPReIDDataModuleStage2
from model.model_pl import CLIPReIDModuleStage1, CLIPReIDModuleStage2
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

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.training.solver.seed)

    output_dir = cfg.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model in the path : {}".format(output_dir))
    
    if DIST_TRAIN:
        strategy = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
            timeout=datetime.timedelta(seconds=1800),
            gradient_as_bucket_view=True,
        )
    else:
        strategy = "auto"
        
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    # Stage 1
    data_module_stage1 = CLIPReIDDataModuleStage1(cfg)
    data_module_stage1.setup()

    dataset_info = data_module_stage1.get_dataset_info()

    model_stage1 = CLIPReIDModuleStage1(
        cfg=cfg,
        num_classes=dataset_info["num_classes"],
        camera_num=dataset_info["camera_num"],
        view_num=dataset_info["view_num"],
    )

    callbacks_stage1 = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename=f"{cfg.model.name}" + "_stage1_" + "-{epoch:02d}-{val_rank1:.4f}",
            monitor="train_loss_stage1",
            mode="min",
            save_top_k=SAVE_TOP_K,
            save_last=SAVE_LAST,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        Timer()
    ]
    
    logger_stage1 = TensorBoardLogger(
        save_dir=output_dir,
        name=cfg.logging.tesorboard_dir,
        default_hp_metric=False,
    )
    
    trainer_stage1 = pl.Trainer(
        max_epochs=cfg.training.solver.stage1.max_epochs,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        strategy=strategy,
        precision=PRECISION,
        callbacks=callbacks_stage1,
        logger=logger_stage1,
        log_every_n_steps=cfg.training.solver.log_period,
        check_val_every_n_epoch=cfg.training.solver.eval_period,
        deterministic=DETERMINISTIC,
    )
    
    model_path = os.path.join(cfg.output_dir, cfg.model.model_chkp_name_stage1)
    if not os.path.exists(model_path):
        print("Not found checkpoint. Start training stage 1")
        trainer_stage1.fit(model_stage1, data_module_stage1)
        model_after_stage2 = model_stage1.model
    else:
        print("Loading from checkpoint {}".format(model_path))
        state_dict = torch.load(model_path, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        model_after_stage2 = model_stage1.model
        model_after_stage2.load_state_dict(new_state_dict, strict=False)
    
    # Stage 2
    data_module_stage2 = CLIPReIDDataModuleStage2(cfg, data_module_stage1.dataset)
    data_module_stage2.setup()

    dataset_info = data_module_stage2.get_dataset_info()

    model_stage2 = CLIPReIDModuleStage2(
        cfg=cfg,
        model=model_after_stage2,
        num_classes=dataset_info["num_classes"],
        num_query=dataset_info["num_query"],
    )

    callbacks_stage2 = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename=f"{cfg.model.name}" + "_stage2_" + "-{epoch:02d}-{val_rank1:.4f}",
            monitor="train_acc_stage2",
            mode="max",
            save_top_k=SAVE_TOP_K,
            save_last=SAVE_LAST,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="train_acc_stage2",
            patience=EARLY_STOPPING_PATIENCE,
            mode=EARLY_STOPPING_MODE,
            verbose=True,
        ),
        Timer()
    ]
    
    logger_stage2 = TensorBoardLogger(
        save_dir=output_dir,
        name=cfg.logging.tesorboard_dir,
        default_hp_metric=False,
    )
    
    trainer_stage2 = pl.Trainer(
        max_epochs=cfg.training.solver.stage2.max_epochs,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        strategy=strategy,
        precision=PRECISION,
        callbacks=callbacks_stage2,
        logger=logger_stage2,
        log_every_n_steps=cfg.training.solver.log_period,
        check_val_every_n_epoch=cfg.training.solver.eval_period,
        deterministic=DETERMINISTIC,
    )
    
    trainer_stage2.fit(model_stage2, data_module_stage2)


if __name__ == "__main__":
    main()