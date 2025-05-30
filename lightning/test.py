import argparse
import os
import torch
import pytorch_lightning as pl

from config import cfg
from lightning.data_module import ReIDDataModule
from lightning.model_module import ReIDModule


def main():
    parser = argparse.ArgumentParser(description="ReID Testing with PyTorch Lightning")
    parser.add_argument(
        "--config_file",
        default="configs/person/vit_clipreid.yml",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to the checkpoint to load",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Create output directory
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize data module
    data_module = ReIDDataModule(cfg)
    data_module.setup()
    
    # Get dataset info
    dataset_info = data_module.get_dataset_info()
    
    # Initialize model module
    model = ReIDModule.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        num_classes=dataset_info["num_classes"],
        camera_num=dataset_info["cam_num"],
        view_num=dataset_info["view_num"],
        num_query=dataset_info["num_query"]
    )
    
    # Set model to stage 2 for testing
    model.stage = 2
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",  # Use mixed precision
    )
    
    # Test model
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
