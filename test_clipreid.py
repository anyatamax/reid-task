import argparse
import os
import torch

from datasets.data import CLIPReIDDataModuleStage1
from model.model_pl import CLIPReIDModuleStage1
from utils.dvc_utils import download_dvc_data
from utils.download_data import download_data
from configs.constants import *

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    model_path = os.path.join(cfg.output_dir, cfg.testing.weight)
    if not os.path.exists(model_path):
        print("Not found result model. Need to train in train_clipreid.py or download from dvc")
        return
    
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
            
    data_module = CLIPReIDDataModuleStage1(cfg)
    data_module.setup()
    
    dataset_info = data_module.get_dataset_info()

    model = CLIPReIDModuleStage1(
        cfg=cfg,
        num_classes=dataset_info["num_classes"],
        camera_num=dataset_info["camera_num"],
        view_num=dataset_info["view_num"],
        num_query=dataset_info["num_query"],
    )
    
    print("Loading from checkpoint {}".format(model_path))
    state_dict = torch.load(model_path, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    model.model.load_state_dict(new_state_dict, strict=False)
    
    logger_test = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        run_name=cfg.logging.run_name + "_predict",
        tracking_uri=cfg.logging.mlflow_tracking_uri,
    )

    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        logger=logger_test,
    )
    
    trainer.predict(model, datamodule=data_module)
    
    cmc, mAP, _, _, _, _, _ = model.evaluator.compute()
    print(f"Test Results - Rank-1: {cmc[0]:.2%}")
    print(f"Test Results - Rank-5: {cmc[4]:.2%}")
    print(f"Test Results - mAP: {mAP:.2%}")

if __name__ == "__main__":
    main()