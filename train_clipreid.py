import random
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)
from pytorch_lightning.loggers import TensorBoardLogger

from configs.constants import (
    ACCELERATOR,
    DETERMINISTIC,
    DEVICE,
    DEVICES,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_PATIENCE,
    PRECISION,
    SAVE_LAST,
    SAVE_TOP_K,
)
from datasets.data import CLIPReIDDataModuleStage0, CLIPReIDDataModuleStage1, CLIPReIDDataModuleStage2
from model.model_pl import CLIPReIDModuleStage0, CLIPReIDModuleStage1, CLIPReIDModuleStage2
from model.onnx_wrapper import CLIPReIDONNXWrapper
from utils.download_data import download_additional_files, download_data
from utils.dvc_utils import download_dvc_data
from utils.export_onnx import export_model_to_onnx


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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

    set_seed(cfg.training.solver.seed)

    output_model_dir = cfg.output_dir
    if output_model_dir:
        output_path = Path(output_model_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
    print("Saving model in the path : {}".format(output_model_dir))

    output_log_dir = cfg.logging.output_log_dir
    if output_log_dir:
        log_path = Path(output_log_dir)
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
    print("Logging in the path : {}".format(output_log_dir))

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

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    model_after_stage0 = None
    if cfg.training.compute_stage0:
        print("Starting Stage0: CLIP Pretraining with Market1501 captions")

        data_module_stage0 = CLIPReIDDataModuleStage0(cfg)
        data_module_stage0.setup()
        
        dataset_info_stage0 = data_module_stage0.get_dataset_info()

        model_stage0 = CLIPReIDModuleStage0(
            cfg,
            num_classes=dataset_info_stage0["num_classes"], 
            camera_num=dataset_info_stage0["camera_num"],
            view_num=dataset_info_stage0["view_num"],
            num_query=dataset_info_stage0["num_query"]
        )
        
        callbacks_stage0 = [
            ModelCheckpoint(
                dirpath=output_model_dir,
                filename=f"{cfg.model.name}"
                + "_stage0"
                + "-{epoch:02d}-{train_loss_stage0:.4f}",
                monitor="train_loss_stage0",
                mode="min",
                save_top_k=SAVE_TOP_K,
                save_last=SAVE_LAST,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            Timer(),
        ]
        
        logger_stage0 = TensorBoardLogger(
            save_dir=output_log_dir,
            name=cfg.logging.experiment_name,
            version=cfg.logging.run_name + "_stage0",
        )
        
        trainer_stage0 = pl.Trainer(
            max_epochs=cfg.training.solver.stage0.max_epochs,
            accelerator=ACCELERATOR,
            devices=DEVICES,
            strategy="auto",
            precision=PRECISION,
            deterministic=DETERMINISTIC,
            callbacks=callbacks_stage0,
            logger=logger_stage0,
            log_every_n_steps=cfg.training.solver.log_period,
            check_val_every_n_epoch=None,
            num_sanity_val_steps=0,
        )
        
        stage0_model_path = Path(cfg.output_dir) / cfg.model.model_chkp_name_stage0
        if not stage0_model_path.exists():
            print("Not found Stage0 checkpoint. Start training Stage 0")
            trainer_stage0.fit(model_stage0, datamodule=data_module_stage0)
            torch.save(model_stage0.model.state_dict(), stage0_model_path)
            model_after_stage0 = model_stage0.model
        else:
            print("Loading from checkpoint {}".format(stage0_model_path))
            state_dict = torch.load(
                stage0_model_path, weights_only=True, map_location=torch.device(DEVICE)
            )
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[name] = v
            model_after_stage0 = model_stage0.model
            model_after_stage0.load_state_dict(new_state_dict, strict=False)

    data_module_stage1 = CLIPReIDDataModuleStage1(cfg)
    data_module_stage1.setup()

    dataset_info = data_module_stage1.get_dataset_info()

    model_stage1 = CLIPReIDModuleStage1(
        cfg=cfg,
        num_classes=dataset_info["num_classes"],
        camera_num=dataset_info["camera_num"],
        view_num=dataset_info["view_num"],
        num_query=dataset_info["num_query"],
        model=model_after_stage0,
    )
    
    if model_after_stage0 is not None:
        print("✅ Stage1 initialized with Stage0 pretrained encoders!")

    callbacks_stage1 = [
        ModelCheckpoint(
            dirpath=output_model_dir,
            filename=f"{cfg.model.name}"
            + "_stage1"
            + "-{epoch:02d}-{train_loss_stage1:.4f}",
            monitor="train_loss_stage1",
            mode="min",
            save_top_k=SAVE_TOP_K,
            save_last=SAVE_LAST,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        Timer(),
    ]

    logger_stage1 = TensorBoardLogger(
        save_dir=output_log_dir,
        name=cfg.logging.experiment_name,
        version=cfg.logging.run_name + "_stage1",
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

    model_path = Path(cfg.output_dir) / cfg.model.model_chkp_name_stage1
    if not model_path.exists():
        print("Not found checkpoint. Start training stage 1")
        trainer_stage1.fit(model_stage1, datamodule=data_module_stage1)
        torch.save(model_stage1.model.state_dict(), model_path)
        model_after_stage2 = model_stage1.model
    else:
        print("Loading from checkpoint {}".format(model_path))
        state_dict = torch.load(
            model_path, weights_only=True, map_location=torch.device(DEVICE)
        )
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

    # if data_module_stage2.use_graph_sampling:
    data_module_stage2.set_model_for_graph_sampling(model_stage2.model)

    callbacks_stage2 = [
        ModelCheckpoint(
            dirpath=output_model_dir,
            filename=f"{cfg.model.name}"
            + "_stage2"
            + "-{epoch:02d}-{train_acc_stage2:.4f}",
            monitor="train_acc_stage2",
            mode="max",
            save_top_k=SAVE_TOP_K,
            save_last=SAVE_LAST,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        # EarlyStopping(
        #     monitor="train_acc_stage2",
        #     patience=EARLY_STOPPING_PATIENCE,
        #     mode=EARLY_STOPPING_MODE,
        #     verbose=True,
        # ),
        Timer(),
    ]

    logger_stage2 = TensorBoardLogger(
        save_dir=output_log_dir,
        name=cfg.logging.experiment_name,
        version=cfg.logging.run_name + "_stage2",
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

    # Saving
    model_final = Path(cfg.output_dir) / cfg.model.model_chkp_name_stage2
    if not model_final.exists():
        print("Not found final model. Start training stage 2")
        trainer_stage2.fit(model_stage2, datamodule=data_module_stage2)
        torch.save(
            model_stage2.model.state_dict(),
            model_final,
        )
    else:
        print("Loading from checkpoint {} final model".format(model_final))
        state_dict = torch.load(
            model_final, weights_only=True, map_location=torch.device(DEVICE)
        )
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        model_after_stage2.load_state_dict(new_state_dict, strict=False)

    if cfg.export_to_onnx:
        print("Exporting model to ONNX format...")

        use_camera = cfg.model.sie_camera
        use_view = cfg.model.sie_view

        onnx_wrapper = CLIPReIDONNXWrapper(
            model=model_stage2.model, use_camera=use_camera, use_view=use_view
        )
        input_shape = (
            1,
            3,
            cfg.preprocessing.size_test[0],
            cfg.preprocessing.size_test[1],
        )

        model_onnx_path = Path(cfg.output_dir) / cfg.model.model_chkp_final_onnx
        export_model_to_onnx(
            model=onnx_wrapper,
            save_path=model_onnx_path,
            input_shape=input_shape,
            verbose=True,
            use_camera=use_camera,
            use_view=use_view,
        )


if __name__ == "__main__":
    main()
