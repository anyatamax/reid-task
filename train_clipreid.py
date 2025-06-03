import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp

from datasets.data import CLIPReIDDataModuleStage1, CLIPReIDDataModuleStage2
from model.model_pl import CLIPReIDModuleStage1, CLIPReIDModuleStage2
from configs.constants import *
from utils.dvc_utils import download_dvc_data
from utils.download_data import download_data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Timer
from pytorch_lightning.loggers import MLFlowLogger
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

    output_model_dir = cfg.output_dir
    if output_model_dir and not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    print("Saving model in the path : {}".format(output_model_dir))
        
    output_log_dir = cfg.logging.output_log_dir
    if output_log_dir and not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)
    print("Logging in the path : {}".format(output_log_dir))
    
    # Download data
    data_path =  os.path.join(cfg.dataset.root_dir, cfg.dataset.data_dir, cfg.dataset.dataset_dir)
    if os.path.exists(data_path):
        print("Dataset already downloaded")
    else:
        if cfg.dataset.from_dvc:
            print("Downloading data from DVC remote...")
            download_success = download_dvc_data(data_dir=cfg.dataset.data_dir, dataset_name=cfg.dataset.dataset_dir + ".dvc")
            if not download_success:
                download_data()
        else:
            print("Downloading data from Google Disk...")
            download_data()
        
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
        num_query=dataset_info["num_query"],
    )

    callbacks_stage1 = [
        ModelCheckpoint(
            dirpath=output_model_dir,
            filename=f"{cfg.model.name}" + "_stage1" + "-{epoch:02d}-{train_loss_stage1:.4f}",
            monitor="train_loss_stage1",
            mode="min",
            save_top_k=SAVE_TOP_K,
            save_last=SAVE_LAST,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        Timer()
    ]
    
    logger_stage1 = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        run_name=cfg.logging.run_name + "_stage1",
        save_dir=output_log_dir,
        tracking_uri=cfg.logging.mlflow_tracking_uri,
    )
    
    trainer_stage1 = pl.Trainer(
        max_epochs=cfg.training.solver.stage1.max_epochs,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        strategy="auto",
        precision=PRECISION,
        callbacks=callbacks_stage1,
        logger=logger_stage1,
        log_every_n_steps=cfg.training.solver.log_period,
        check_val_every_n_epoch=cfg.training.solver.stage1.max_epochs + 1,
        deterministic=DETERMINISTIC,
    )
    
    model_path = os.path.join(cfg.output_dir, cfg.model.model_chkp_name_stage1)
    if not os.path.exists(model_path):
        print("Not found checkpoint. Start training stage 1")
        trainer_stage1.fit(model_stage1, datamodule=data_module_stage1)
        torch.save(model_stage1.model.state_dict(), model_path)
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
            dirpath=output_model_dir,
            filename=f"{cfg.model.name}" + "_stage2" + "-{epoch:02d}-{train_acc_stage2:.4f}",
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
    
    logger_stage2 = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        run_name=cfg.logging.run_name + "_stage2",
        save_dir=output_log_dir,
        tracking_uri=cfg.logging.mlflow_tracking_uri,
    )
    
    trainer_stage2 = pl.Trainer(
        max_epochs=cfg.training.solver.stage2.max_epochs,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        strategy="auto",
        precision=PRECISION,
        callbacks=callbacks_stage2,
        logger=logger_stage2,
        log_every_n_steps=cfg.training.solver.log_period,
        check_val_every_n_epoch=None,
        deterministic=DETERMINISTIC,
        num_sanity_val_steps=0,
    )
    
    trainer_stage2.fit(model_stage2, datamodule=data_module_stage2)

    torch.save(
        model_stage2.model.state_dict(),
        os.path.join(cfg.output_dir, cfg.model.model_chkp_name_stage2),
    )

if __name__ == "__main__":
    main()
