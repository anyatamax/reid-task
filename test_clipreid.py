from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from configs.constants import ACCELERATOR, DEVICE, DEVICES, PRECISION
from datasets.data import CLIPReIDDataModuleStage1
from model.model_pl import CLIPReIDModuleStage1
from utils.download_data import download_additional_files, download_data, download_model
from utils.dvc_utils import download_dvc_data


def download(
    root_dir, data_dir, dataset_dir, download_dvc, download_from_disk, from_dvc
):
    data_path = Path(root_dir) / data_dir / dataset_dir
    if data_path.exists():
        print(f"{dataset_dir} already downloaded")
    else:
        if from_dvc:
            print(f"Downloading {dataset_dir} from DVC remote...")
            download_success = download_dvc(
                data_dir=data_dir, dataset_name=dataset_dir + ".dvc"
            )
            if not download_success:
                download_from_disk()
        else:
            print(f"Downloading {dataset_dir} from Google Disk...")
            download_from_disk()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model_path = Path(cfg.output_dir) / cfg.testing.weight
    if not model_path.exists():
        print(
            "Not found result model. Need to train in train_clipreid.py or download from dvc or download from article"
        )
        if cfg.testing.load_from_article:
            model_path = Path(cfg.output_dir) / cfg.testing.article_name_weight
            if not model_path.exists():
                download_model()
        else:
            return

    # Download data
    download(
        cfg.dataset.root_dir,
        cfg.dataset.data_dir,
        cfg.dataset.dataset_dir,
        download_dvc_data,
        download_data,
        cfg.dataset.from_dvc,
    )

    # Download additional files
    download(
        cfg.dataset.root_dir,
        cfg.dataset.data_dir,
        cfg.dataset.files_dir,
        download_dvc_data,
        download_additional_files,
        cfg.dataset.from_dvc,
    )

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
    state_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device(DEVICE)
    )
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    model.model.load_state_dict(new_state_dict, strict=False)

    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
    )

    trainer.predict(model, datamodule=data_module)

    cmc, mAP, _, _, _, _, _ = model.evaluator.compute()
    print(f"Test Results - Rank-1: {cmc[0]:.2%}")
    print(f"Test Results - Rank-5: {cmc[4]:.2%}")
    print(f"Test Results - mAP: {mAP:.2%}")


if __name__ == "__main__":
    main()
