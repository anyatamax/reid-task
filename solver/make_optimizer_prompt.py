import torch


def make_optimizer_stage0(cfg, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.training.solver.stage0.base_lr
        weight_decay = cfg.training.solver.stage0.weight_decay

        if "image_encoder" in key:
            lr = cfg.training.solver.stage0.image_encoder_lr

        elif any(component in key for component in ["transformer", "positional_embedding", "ln_final", "text_projection"]):
            lr = cfg.training.solver.stage0.text_encoder_lr

        elif "token_embedding" in key:
            lr = cfg.training.solver.stage0.token_embedding_lr

        if "bias" in key:
            lr = lr * cfg.training.solver.stage0.bias_lr_factor
            weight_decay = cfg.training.solver.stage0.weight_decay_bias
            
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    if cfg.training.solver.stage0.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.training.solver.stage0.base_lr,
            weight_decay=cfg.training.solver.stage0.weight_decay,
        )
    elif cfg.training.solver.stage0.optimizer_name == "SGD":
        optimizer = getattr(torch.optim, cfg.training.solver.stage0.optimizer_name)(
            params, momentum=cfg.training.solver.stage0.momentum
        )
    else:
        optimizer = getattr(torch.optim, cfg.training.solver.stage0.optimizer_name)(
            params
        )
    
    print(f"Stage0 optimizer created with {len(params)} parameter groups")
    print(f"Keys being optimized STAGE 0: ")
    for key in keys:
        print(key)
    
    return optimizer


def make_optimizer_1stage(cfg, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = cfg.training.solver.stage1.base_lr
            weight_decay = cfg.training.solver.stage1.weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
    if cfg.training.solver.stage1.optimizer_name == "SGD":
        optimizer = getattr(torch.optim, cfg.training.solver.stage1.optimizer_name)(
            params, momentum=cfg.training.solver.stage1.momentum
        )
    elif cfg.training.solver.stage1.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.training.solver.stage1.base_lr,
            weight_decay=cfg.training.solver.stage1.weight_decay,
        )
    else:
        optimizer = getattr(torch.optim, cfg.training.solver.stage1.optimizer_name)(
            params
        )
    
    print(f"Keys being optimized STAGE 1: ")
    for key in keys:
        print(key)

    return optimizer


def make_optimizer_2stage(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = cfg.training.solver.stage2.base_lr
        weight_decay = cfg.training.solver.stage2.weight_decay
        if "bias" in key:
            lr = (
                cfg.training.solver.stage2.base_lr
                * cfg.training.solver.stage2.bias_lr_factor
            )
            weight_decay = cfg.training.solver.stage2.weight_decay_bias
        if cfg.training.solver.stage2.large_fc_lr:
            if "classifier" in key or "arcface" in key:
                lr = cfg.training.solver.stage2.base_lr * 2
                print("Using two times learning rate for fc ")

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.training.solver.stage2.optimizer_name == "SGD":
        optimizer = getattr(torch.optim, cfg.training.solver.stage2.optimizer_name)(
            params, momentum=cfg.training.solver.stage2.momentum
        )
    elif cfg.training.solver.stage2.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.training.solver.stage2.base_lr,
            weight_decay=cfg.training.solver.stage2.weight_decay,
        )
    else:
        optimizer = getattr(torch.optim, cfg.training.solver.stage2.optimizer_name)(
            params
        )
    optimizer_center = torch.optim.SGD(
        center_criterion.parameters(), lr=cfg.training.solver.stage2.center_lr
    )

    print(f"Keys being optimized STAGE 2: ")
    for key in keys:
        print(key)

    return optimizer, optimizer_center
