# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
import operator
from datetime import date

import torch
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from data.data_loader import build_data_loader

from utils.config import setup
import utils.saver as saver
from utils.progress import AverageMeter, ProgressMeter, accuracy
import utils.comm as comm
import utils.logging as logging
from evaluate import attentive_nas_eval as attentive_nas_eval
from sampler.attentive_nas_sampler import ArchSampler as ArchSampler
from solver import build_optimizer, build_lr_scheduler
import utils.loss_ops as loss_ops 
import models
from copy import deepcopy
import numpy as np
import joblib 

from sklearn.ensemble import RandomForestRegressor


parser = argparse.ArgumentParser(description='AttentiveNAS Training')
parser.add_argument('--config-file', default=None, type=str, 
                    help='training configuration')
parser.add_argument('--machine-rank', default=0, type=int, 
                    help='machine rank, distributed setting')
parser.add_argument('--num-machines', default=1, type=int, 
                    help='number of nodes, distributed setting')
parser.add_argument('--dist-url', default="tcp://127.0.0.1:10001", type=str, 
                    help='init method, distributed setting')

logger = logging.get_logger(__name__)


def build_args_and_env(run_args):

    assert run_args.config_file and os.path.isfile(run_args.config_file), 'cannot locate config file'
    args = setup(run_args.config_file)
    args.config_file = run_args.config_file

    #load config
    assert args.distributed and args.multiprocessing_distributed, 'only support DDP training'
    args.distributed = True

    args.machine_rank = run_args.machine_rank
    args.num_nodes = run_args.num_machines
    args.dist_url = run_args.dist_url
    args.models_save_dir = os.path.join(args.models_save_dir, args.exp_name)

    if not os.path.exists(args.models_save_dir):
        os.makedirs(args.models_save_dir)

    #backup config file
    saver.copy_file(args.config_file, '{}/{}'.format(args.models_save_dir, os.path.basename(args.config_file)))

    args.checkpoint_save_path = os.path.join(
        args.models_save_dir, 'attentive_nas.pth.tar'
    )
    args.logging_save_path = os.path.join(
        args.models_save_dir, f'stdout.log'
    )
    return args


def main():
    run_args = parser.parse_args()
    args = build_args_and_env(run_args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    #cudnn.deterministic = True
    #warnings.warn('You have chosen to seed training. '
    #                'This will turn on the CUDNN deterministic setting, '
    #                'which can slow down your training considerably! '
    #                'You may see unexpected behavior when restarting '
    #                'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.num_nodes
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError
    
    assert args.world_size > 1, 'only support ddp training'


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu  # local rank, local machine cuda id
    args.local_rank = args.gpu
    args.batch_size = args.batch_size_per_gpu
    args.batch_size_total = args.batch_size * args.world_size
    #rescale base lr
    args.lr_scheduler.base_lr = args.lr_scheduler.base_lr * (max(1, args.batch_size_total // 256))

    # set random seed, make sure all random subgraph generated would be the same
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)

    global_rank = args.gpu + args.machine_rank * ngpus_per_node
    dist.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=global_rank
    )

    # Setup logging format.
    logging.setup_logging(args.logging_save_path, 'w')

    logger.info(f"Use GPU: {args.gpu}, machine rank {args.machine_rank}, num_nodes {args.num_nodes}, \
                    gpu per node {ngpus_per_node}, world size {args.world_size}")

    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    args.rank = comm.get_rank() # global rank
    args.local_rank = args.gpu
    torch.cuda.set_device(args.gpu)

    # build model
    logger.info("=> creating model '{}'".format(args.arch))
    model = models.model_factory.create_model(args)
    model.cuda(args.gpu)

    #build arch sampler
    arch_sampler = None
    if getattr(args, 'sampler', None):
        arch_sampler = ArchSampler(
            args.sampler.arch_to_flops_map_file_path, args.sampler.discretize_step, model, None 
        )

    # use sync batchnorm
    if getattr(args, 'sync_bn', False):
        model.apply(
                lambda m: setattr(m, 'need_sync', True))

    model = comm.get_parallel_model(model, args.gpu) #local rank

    logger.info(model)

    criterion = loss_ops.CrossEntropyLossSmooth(args.label_smoothing).cuda(args.gpu)
    soft_criterion = loss_ops.KLLossSoft().cuda(args.gpu)

    if not getattr(args, 'inplace_distill', True):
        soft_criterion = None

    ## load dataset, train_sampler: distributed
    train_loader, val_loader, train_sampler =  build_data_loader(args)
    args.n_iters_per_epoch = len(train_loader)

    logger.info( f'building optimizer and lr scheduler, \
            local rank {args.gpu}, global rank {args.rank}, world_size {args.world_size}')
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
 
    # optionally resume from a checkpoint
    if args.resume:
        saver.load_checkpoints(args, model, optimizer, lr_scheduler, logger)

    logger.info(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        args.curr_epoch = epoch
        logger.info('Training lr {}'.format(lr_scheduler.get_lr()[0]))

        # train for one epoch
        acc1, acc5 = train_epoch(epoch, model, train_loader, optimizer, criterion, args, \
                arch_sampler=arch_sampler, soft_criterion=soft_criterion, lr_scheduler=lr_scheduler)

        if comm.is_master_process() or args.distributed:
            # validate supernet model
            validate(
                train_loader, val_loader, model, criterion, args
            )

        if comm.is_master_process():
            # save checkpoints
            saver.save_checkpoint(
                args.checkpoint_save_path, 
                model,
                optimizer,
                lr_scheduler, 
                args,
                epoch,
            )


def train_epoch(
    epoch, 
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    args, 
    arch_sampler=None,
    soft_criterion=None, 
    lr_scheduler=None, 
):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()

    num_updates = epoch * len(train_loader)

    for batch_idx, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # total subnets to be sampled
        num_subnet_training = max(2, getattr(args, 'num_arch_training', 2))
        optimizer.zero_grad()

        ### compute gradients using sandwich rule ###
        # step 1 sample the largest network, apply regularization to only the largest network
        drop_connect_only_last_two_stages = getattr(args, 'drop_connect_only_last_two_stages', True)
        model.module.sample_max_subnet()
        model.module.set_dropout_rate(args.dropout, args.drop_connect, drop_connect_only_last_two_stages) #dropout for supernet
        output = model(images)
        loss = criterion(output, target)
        loss.backward()

        with torch.no_grad():
            soft_logits = output.clone().detach()

        #step 2. sample the smallest network and several random networks
        sandwich_rule = getattr(args, 'sandwich_rule', True)
        model.module.set_dropout_rate(0, 0, drop_connect_only_last_two_stages)  #reset dropout rate
        for arch_id in range(1, num_subnet_training):
            if arch_id == num_subnet_training-1 and sandwich_rule:
                model.module.sample_min_subnet()
            else:
                # attentive sampling with training loss as the surrogate performance metric 
                if arch_sampler is not None:
                    sampling_method = args.sampler.method
                    if sampling_method in ['bestup', 'worstup']:
                        target_flops = arch_sampler.sample_one_target_flops()
                        candidate_archs = arch_sampler.sample_archs_according_to_flops(
                            target_flops, n_samples=args.sampler.num_trials
                        )
                        my_pred_accs = []
                        for arch in candidate_archs:
                            model.module.set_active_subnet(**arch)
                            with torch.no_grad():
                                my_pred_accs.append(-1.0 * criterion(model(images), target))

                        if sampling_method == 'bestup':
                            idx, _ = max(enumerate(my_pred_accs), key=operator.itemgetter(1))                          
                        else:
                            idx, _ = min(enumerate(my_pred_accs), key=operator.itemgetter(1))                          
                        model.module.set_active_subnet(**candidate_archs[idx])  #reset
                    else:
                        raise NotImplementedError
                else:
                    model.module.sample_active_subnet()

            # calcualting loss
            output = model(images)

            if soft_criterion:
                loss = soft_criterion(output, soft_logits)
            else:
                assert not args.inplace_distill
                loss = criterion(output, target)

            loss.backward()

        #clip gradients if specfied
        if getattr(args, 'grad_clip_value', None):
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)

        optimizer.step()

        #accuracy measured on the local batch
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if args.distributed:
            corr1, corr5, loss = acc1*args.batch_size, acc5*args.batch_size, loss.item()*args.batch_size #just in case the batch size is different on different nodes
            stats = torch.tensor([corr1, corr5, loss, args.batch_size], device=args.gpu)
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM) 
            corr1, corr5, loss, batch_size = stats.tolist()
            acc1, acc5, loss = corr1/batch_size, corr5/batch_size, loss/batch_size
            losses.update(loss, batch_size)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
        else:
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        num_updates += 1
        if lr_scheduler is not None:
            lr_scheduler.step()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx, logger)

    return top1.avg, top5.avg


def validate(
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    args, 
    distributed = True,
):
    subnets_to_be_evaluated = {
        'attentive_nas_min_net': {},
        'attentive_nas_max_net': {},
    }

    acc1_list, acc5_list = attentive_nas_eval.validate(
        subnets_to_be_evaluated,
        train_loader,
        val_loader, 
        model, 
        criterion,
        args,
        logger,
        bn_calibration = True,
    )



if __name__ == '__main__':
    main()


