# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR, WarmupLinearDecayLR, ConstantLR

def build_optimizer(args, model):
    """
        Build an optimizer from config.
    """
    no_wd_params, wd_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if ".bn" in name or ".bias" in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)
    no_wd_params = nn.ParameterList(no_wd_params)
    wd_params = nn.ParameterList(wd_params)

    weight_decay_weight = args.weight_decay_weight
    weight_decay_bn_bias = args.weight_decay_bn_bias
    base_lr = args.lr_scheduler.base_lr

    params_group = [
        {"params": wd_params, "weight_decay": float(weight_decay_weight), 'group_name':'weight'},
        {"params": no_wd_params, "weight_decay": float(weight_decay_bn_bias), 'group_name':'bn_bias'},
    ]

    if args.optimizer.method == 'sgd':
        momentum = args.optimizer.momentum
        nesterov = args.optimizer.nesterov
        optimizer = torch.optim.SGD(
            params_group,
            lr = base_lr,
            momentum = momentum,
            nesterov = nesterov,
        )
    else:
        raise ValueError(f'no optimizer {args.optimizer.method}')
    
    return optimizer


def build_lr_scheduler(args, optimizer):

    if not hasattr(args, 'max_iters'):
        #important house-keeping stuff
        args.max_iters = args.n_iters_per_epoch * args.epochs
        
    if getattr(args, 'warmup_iters', None) is None:
        args.warmup_iters = args.n_iters_per_epoch * args.warmup_epochs

    warmup_iters = args.warmup_iters
    warmup_lr = float(getattr(args.lr_scheduler, 'warmup_lr', 0.001))
    warmup_method = getattr(args.lr_scheduler, 'warmup_method', 'linear')
    clamp_lr_percent = float(getattr(args.lr_scheduler, 'clamp_lr_percent', 0.))
    clamp_lr = args.lr_scheduler.base_lr * clamp_lr_percent

    if args.lr_scheduler.method == 'warmup_cosine_lr':
        return WarmupCosineLR(
                optimizer, 
                args.max_iters, 
                warmup_factor = warmup_lr,
                warmup_iters = warmup_iters,
                warmup_method = warmup_method,
                clamp_lr = clamp_lr,
        )
    elif args.lr_scheduler.method == 'warmup_exp_decay_lr':
        decay_cycle_iters = int(args.lr_scheduler.lr_decay_cycle * args.n_iters_per_epoch)
        total_decay_iters = args.n_iters_per_epoch * (args.epochs - args.warmup_epochs)
        milestones = [ warmup_iters + (idx + 1) * decay_cycle_iters \
                        for idx in range(total_decay_iters // decay_cycle_iters)]
        return WarmupMultiStepLR(
                optimizer,
                milestones,
                gamma=args.lr_scheduler.lr_decay_rate_per_cycle,
                warmup_factor = warmup_lr,
                warmup_iters = warmup_iters,
                warmup_method = warmup_method,
                clamp_lr = clamp_lr,
        )
    elif args.lr_scheduler.method == 'warmup_linear_lr':
        decay_cycle_iters = args.n_iters_per_epoch
        milestones = [ warmup_iters + (idx + 1) * decay_cycle_iters \
                        for idx in range(args.epochs - args.warmup_epochs)]
        return WarmupLinearDecayLR(
                optimizer, 
                milestones,
                warmup_factor = warmup_lr,
                warmup_iters = warmup_iters,
                warmup_method = warmup_method,
                clamp_lr = clamp_lr,
        )
    elif args.lr_scheduler.method == 'constant_lr':
        return ConstantLR(
                optimizer
        )
    else:
        raise NotImplementedError

