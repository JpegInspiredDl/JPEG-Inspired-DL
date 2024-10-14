import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
import sys

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from engine import train_one_epoch, evaluate

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils as utils

import models.convnext
import models.vision_transformer

from datasets import build_dataset

from torch import optim as optim
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


module_path = '/home/ahamsala/PROJECT_AH/JPEG_DNN'
if module_path not in sys.path: sys.path.append(module_path)
from helper.JPEG_layer import *

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=50, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--initialize', default='',
                        help='initialize from a model file')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET',
                        type=str, help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    # JPEG layer

    parser.add_argument('--ADAM_enable', action='store_true')
    parser.add_argument('--JPEG_enable', action='store_true')
    parser.add_argument('--JPEG_alpha_trainable', action='store_true')
    parser.add_argument('--JPEG_Layer_lr_scheduler_cos', action='store_true')
    parser.add_argument('--hardness_matching', action='store_true')

    

    parser.add_argument('--zero_dc', action='store_true')
    parser.add_argument('--alpha_scaling', action='store_true')
    parser.add_argument('--alpha_fixed', action='store_true')
    
    parser.add_argument('--initial_Q_w_sensitivity', action='store_true')
    parser.add_argument('--initial_Q_w_abs_avg', action='store_true')

    parser.add_argument('--JEPG_alpha', type=float, default=1.0, help='Tempurature scaling')
    parser.add_argument('--JEPG_learning_rate', type=float, default=0.0125, help='Quantization Table Learning Rate')
    parser.add_argument('--alpha_learning_rate', type=float, default=None, help='Alpha Learning Rate')
    parser.add_argument('--Q_inital', type=float, default=1.0, help='Initial Quantization Step')
    parser.add_argument('--block_size', type=int, default=8, help='the experiment id')
    parser.add_argument('--num_bit', type=int, default=8, help='Number of bits to represent DCT coeff')
    parser.add_argument('--min_Q_Step', type=float, default=0.0, choices=range(0, 256), help='Minumum Quantization Step')
    parser.add_argument('--max_Q_Step', type=float, default=255, choices=range(0, 256), help='Maximum Quantization Step')
    parser.add_argument('--num_non_zero_q', type=int, default=5, choices=range(2, 2**11), help='Window size for the reconstruction space')
    parser.add_argument('--hardness', type=float, default=2, help='Hardness of the quantizer')
    parser.add_argument('--std_width_Y', type=float, default=1, help='Width of the standard deviation')
    parser.add_argument('--std_width_CbCr', type=float, default=1, help='Width of the standard deviation')
    parser.add_argument('--jpeg_layers', type=int, default=1, help='Number of JPEG layers')

    parser.add_argument('--p_quantile', type=float, default=0.2, help='Width of the standard deviation')
    
    parser.add_argument('--log_add', type=str, default='', help='add text to the file')
    parser.add_argument('--q_max', type=float, default=10, help='maximum q in the initial quantizaiton table')
    parser.add_argument('--hardness_th', type=float, default=2.5, help='')
    
    parser.add_argument("--lr-step-size", default=100, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.05, type=float, help="decrease lr by a factor of lr-gamma")
    
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # dataloader
    mean = IMAGENET_INCEPTION_MEAN if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    
    if args.JPEG_enable:
        mean_datatloader=(0, 0, 0)
        std_datatloader=(1/255., 1/255., 1/255.)
    else:
        mean_datatloader=mean
        std_datatloader=std

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args, 
                                        mean=mean_datatloader, std=std_datatloader)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args, 
                                        mean=mean_datatloader, std=std_datatloader)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    
    
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    underlying_model = utils.build_model(args)

    '''
    Here we add the model
    '''

    args.dir_initial_abs_avg = "../Initialization_alpha_Q_tables/{}_abs_avg.pt".format(args.data_set) 
    args.sensitvity_dir = "/home/ahamsala/PROJECT_AH/JPEG_DNN/Senstivity/Senstivity_{}/SenMap/".format(args.data_set)
    samples, _ = next(iter(data_loader_train))
    img_shape = samples.shape[2:]
    jpeg_layer = JPEG_layer(opt=args, img_shape=img_shape, mean=mean, std=std)        
    model = CustomModel(jpeg_layer, underlying_model)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    if args.zero_dc:
        optimizer_JPEG_Layer = None
        JPEG_layer_lr_schedule_values = None
    
    elif args.JPEG_enable:
        # ADAM + cosine scheduler
        optimizer_JPEG_layer_data = [
                            {'params': model_without_ddp.jpeg_layer.lum_qtable, 'lr': args.JEPG_learning_rate,},
                            {'params': model_without_ddp.jpeg_layer.chrom_qtable, 'lr': args.JEPG_learning_rate,},
                            ]
        optimizer_JPEG_Layer = optim.Adam(optimizer_JPEG_layer_data)
        # JPEG_layer_lr_schedule_values = utils.cosine_scheduler(
        #                                     args.JEPG_learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        #                                     warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        #                                 )

        # args.JPEG_Layer_lr_scheduler_cos = False
        # SGD + linear learning rate decaying
        # optimizer_JPEG_layer_data = [
        #                     {'params':  model_without_ddp.jpeg_layer.lum_qtable, 'lr': args.JEPG_learning_rate, 'momentum': args.momentum,},
        #                     {'params': model_without_ddp.jpeg_layer.chrom_qtable, 'lr': args.JEPG_learning_rate, 'momentum': args.momentum,},
        #                  ]

        # optimizer_JPEG_Layer = optim.SGD(optimizer_JPEG_layer_data)
        # JPEG_layer_lr_schedule_values = torch.optim.lr_scheduler.StepLR(optimizer_JPEG_Layer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        JPEG_layer_lr_schedule_values = None

    
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    log_fname = os.path.join(args.output_dir, 'log_JPEG.txt')
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start_time_inner = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        
        if args.JPEG_enable and args.gpu == 0:
            lum_qtable =  model_without_ddp.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            chrom_qtable =  model_without_ddp.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0)
            alpha_lum =  model_without_ddp.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alpha_chrom =  model_without_ddp.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alpha = torch.cat((alpha_lum, alpha_chrom), 0)
            hardness = alpha * (quantizationTable)**2
            
            print("Hardness --> (Num# of Coeff > {}) Y: {:.3f}, CbCr: {:.3f}".format(args.hardness_th, (hardness[0]>args.hardness_th).sum(), (hardness[1]>args.hardness_th).sum()))
            print("Hardness --> Min: {:.3f}, Max: {:.3f}".format(hardness.min().item(), hardness.max().item()))
            print("Q Table  --> Min: {:.3f}, Max: {:.2f}".format(quantizationTable.min().item(), quantizationTable.max().item()))
            print("Alpha    --> Min: {:.3f}, Max: {:.3f}".format(alpha.min().item(), alpha.max().item()))
            

            str_q_alpha = "| {:.3f} \t {:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t".format(
                                                                            args.hardness_th,
                                                                            (hardness[0]>args.hardness_th).sum(), (hardness[1]>args.hardness_th).sum(),
                                                                            hardness.min().item(), hardness.max().item(),
                                                                            quantizationTable.min().item(), quantizationTable.max().item(),
                                                                            alpha.min().item(), alpha.max().item(),
                                                                        )
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp, args=args, optimizer_JPEG_Layer=optimizer_JPEG_Layer, JPEG_layer_lr_schedule_values=JPEG_layer_lr_schedule_values,
        )




        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            
                if args.gpu == 0:
                    QF_fname = os.path.join(args.output_dir, 'q_table_best.pt')
                    alpha_fname = os.path.join(args.output_dir, 'alpha_table_best.pt')
                    lum_qtable =  model_without_ddp.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                    chrom_qtable =  model_without_ddp.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                    quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0).cpu()
                    torch.save(quantizationTable, QF_fname)

                    alpha_lum =  model_without_ddp.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                    alpha_chrom =  model_without_ddp.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                    alpha = torch.cat((alpha_lum, alpha_chrom), 0).cpu()
                    torch.save(alpha, alpha_fname)

            total_time_inner = time.time() - start_time_inner
            
            if args.gpu == 0:
                with open(log_fname, 'a+') as log:
                    log.write(str(epoch) + "\t" +str(test_stats["acc1"])+'\t'+ str(max_accuracy) + '\t' +  
                              str(datetime.timedelta(seconds=int(total_time_inner))) + '\t' + str_q_alpha + '\n')

            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
                log_writer.update(best_acc=max_accuracy, head="perf", step=epoch)

            log_stats = {'best_acc': max_accuracy,
                         **{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    if args.gpu == 0:
        exp_txt = open('experimetal_Results_JPEG_DL.txt', 'a+')
        exp_txt.write(args.output_dir +"\t"+ str(epoch) + "\t" + str(max_accuracy) + "\t" + total_time_str + "\n") # Write some text
        exp_txt.close() # Close the file
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)