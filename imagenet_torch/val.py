import datetime
import os
import sys
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix
from torchvision import models

# module_path = '/home/ahamsala/PROJECT_AH/JPEG_DNN'
# if module_path not in sys.path: sys.path.append(module_path)
# from helper.JPEG_layer import *

# from JPEG_layer import *
# from JPEG_layer_standard import *
from utils import load_model_imagenet

import socket

hostname = socket.gethostname()

warnings.filterwarnings('ignore')

# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

# def load_model_imagenet(name, pretrained=False, **kwargs):
#     if name=="resnet18":
#         model = models.resnet18(pretrained=pretrained)
#     elif name=="resnet34":
#         model = models.resnet34(pretrained=pretrained)
#     elif name=="resnet50":
#         model = models.resnet50(pretrained=pretrained)
#     elif name=="resnet101":
#         model = models.resnet101(pretrained=pretrained)
#     elif name=="resnet152":
#         model = models.resnet152(pretrained=pretrained)
#     return model

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, optimizer_JPEG_Layer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if args.JPEG_enable:
            optimizer_JPEG_Layer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        
        if args.JPEG_enable:
            optimizer_JPEG_Layer.step()
        
        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        # break



def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        if args.JPEG_enable:
            mean=(0, 0, 0)
            std=(1/255., 1/255., 1/255.)
        else:
            mean=(0.485, 0.456, 0.406)
            std=(0.229, 0.224, 0.225)
        
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
                mean=mean,
                std=std,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            if "emsolserv" in hostname:
                preprocessing = weights.transforms()
            else:
                preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,                
                mean=mean,
                std=std,
            )


        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.JPEG_mode == "Single_Q_table":
        from JPEG_layer_standard import JPEG_layer, CustomModel
    elif args.JPEG_mode == "Multi_Q_table":
        from JPEG_layer import JPEG_layer, CustomModel
    else:
        raise RuntimeError('choose between Single_Q_table or Multi_Q_table Modes')

    if "emsolserv" in hostname:
        args.data_path = "/home/ahamsala/datasets/imagenet/"
    elif "multicomgpu.eng" in hostname:
        args.data_path = "/home/l44ye/datasets/"
    else:
        print("The current directory is ", args.data_path)
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    args.dir_initial_std = "../Initialization_alpha_Q_tables/{}_std.pt".format(args.dataset) 
    args.dir_initial_mean = "../Initialization_alpha_Q_tables/{}_mean.pt".format(args.dataset) 
    args.dir_initial_abs_avg = "../Initialization_alpha_Q_tables/{}_abs_avg.pt".format(args.dataset) 
    if "Single_Q_table" in args.output_dir:
        args.sensitvity_dir = "../Senstivity/Senstivity_{}/SenMap/".format(args.dataset)
    elif "Multi_Q_table" in args.output_dir:
        args.sensitvity_dir = "../Senstivity/Senstivity_{}/SenMap_Multi_Q/".format(args.dataset)
    else:
        args.sensitvity_dir = "../Senstivity/Senstivity_{}/SenMap/".format(args.dataset)
    
    if args.max_Q_Step == 0:
        args.max_Q_Step = None

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    if "emsolserv" in hostname:
        underlying_model = load_model_imagenet(args.model, num_classes=num_classes)
    else:
        underlying_model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)



    if args.JPEG_enable:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # samples, _ = next(iter(data_loader_train))
        # img_shape = samples.shape[2:]
        jpeg_layer = JPEG_layer(opt=args, img_shape=(args.train_crop_size, args.train_crop_size, 3), mean=mean, std=std)
        model = CustomModel(jpeg_layer, underlying_model)
    else:
        model = underlying_model
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if "emsolserv" in hostname:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    
    if args.JPEG_enable:
        parameters = utils.set_weight_decay(
            model.underlying_model,
            args.weight_decay,
            norm_weight_decay=args.norm_weight_decay,
            custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
        )
    else:
        parameters = utils.set_weight_decay(
            model,
            args.weight_decay,
            norm_weight_decay=args.norm_weight_decay,
            custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
        )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    optimizer_JPEG_Layer = None
    
    if args.JPEG_enable:
        args.alpha_learning_rate = args.JEPG_learning_rate
        if args.optimizer == "SGD":
            optimizer_JPEG_layer_data = [
                                {'params': model.jpeg_layer.lum_qtable, 'lr': args.JEPG_learning_rate, 'momentum': args.momentum,},
                                {'params': model.jpeg_layer.chrom_qtable, 'lr': args.JEPG_learning_rate, 'momentum': args.momentum,},
                                ]
            if args.JPEG_alpha_trainable:
                optimizer_JPEG_layer_data.append({'params': model.jpeg_layer.alpha_lum, 'lr': args.alpha_learning_rate, 'momentum': args.momentum,})
                optimizer_JPEG_layer_data.append({'params': model.jpeg_layer.alpha_chrom, 'lr': args.alpha_learning_rate, 'momentum': args.momentum,})
 

            optimizer_JPEG_Layer = torch.optim.SGD(optimizer_JPEG_layer_data)
        elif args.optimizer == "ADAM":
            optimizer_JPEG_layer_data = [
                                {'params': model.jpeg_layer.lum_qtable, 'lr': args.JEPG_learning_rate,},
                                {'params': model.jpeg_layer.chrom_qtable, 'lr': args.JEPG_learning_rate, },
                                ]
            
            if args.JPEG_alpha_trainable:
                optimizer_JPEG_layer_data.append({'params': model.jpeg_layer.alpha_lum, 'lr': args.alpha_learning_rate, })
                optimizer_JPEG_layer_data.append({'params': model.jpeg_layer.alpha_chrom, 'lr': args.alpha_learning_rate, })
 
            optimizer_JPEG_Layer = torch.optim.Adam(optimizer_JPEG_layer_data)
        else:
            raise RuntimeError('Optimizer is not selected')

    
        if args.JPEG_Layer_lr_scheduler_step or  args.optimizer == 'SGD':
            JPEG_Layer_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_JPEG_Layer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.JPEG_Layer_lr_scheduler_cos:
            JPEG_Layer_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_JPEG_Layer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min)
        else:
            JPEG_Layer_lr_scheduler = None
            Exception('lr_scheduler is not selected')
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
    


    
    if args.resume:
        # checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        
        QF_fname = os.path.join(args.output_dir, 'q_table.pt')

        lum_qtable =  model_without_ddp.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        chrom_qtable =  model_without_ddp.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0).cpu()
        torch.save(quantizationTable, QF_fname)
        return

    print("Start training")
    best_accuracy = 0

    start_time = time.time()
    log_fname = os.path.join(args.output_dir, 'log.txt')

    for epoch in range(args.start_epoch, args.epochs):
        start_time_inner = time.time()
        
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
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, optimizer_JPEG_Layer)
        
        lr_scheduler.step()
        if args.JPEG_enable and JPEG_Layer_lr_scheduler is not None:
            JPEG_Layer_lr_scheduler.step()
        
        tp1_accuracy = evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                'optimizer_JPEG_layer': optimizer_JPEG_Layer.state_dict(),
                'accuracy': tp1_accuracy,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint_last.pth"))
            
            if best_accuracy < tp1_accuracy:
                best_accuracy = tp1_accuracy

                if args.gpu == 0:
                    utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_best.pth"))
                    print('best accuracy:', best_accuracy)
                    
                    QF_fname = os.path.join(args.output_dir, 'q_table.pt')
                    alpha_fname = os.path.join(args.output_dir, 'alpha_table.pt')

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
                log.write(str(epoch) + "\t" +str(tp1_accuracy)+'\t'+ str(best_accuracy) + '\t' +  str(datetime.timedelta(seconds=int(total_time_inner))) + '\t' + str_q_alpha + '\n')

        # break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if args.gpu == 0:
        exp_txt = open('experimetal_Results_JPEG_DL.txt', 'a+')
        exp_txt.write(args.output_dir +"\t"+ str(epoch) + "\t" + str(best_accuracy) + "\t" + total_time_str + "\n") # Write some text
        exp_txt.close() # Close the file

        print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument('--dataset', type=str, default='ImageNet', choices=['CUB200', 'ImageNet'], help='dataset')
    
    parser.add_argument('--ADAM_enable', action='store_true')
    parser.add_argument('--JPEG_enable', action='store_true')

    
    parser.add_argument('--JPEG_alpha_trainable', action='store_true')

    parser.add_argument('--zero_dc', action='store_true')
    parser.add_argument('--alpha_scaling', action='store_true')
    parser.add_argument('--alpha_fixed', action='store_true')
    parser.add_argument('--JPEG_Layer_lr_scheduler_cos', action='store_true')
    parser.add_argument('--JPEG_Layer_lr_scheduler_step', action='store_true')
    
    parser.add_argument('--initial_Q_w_sensitivity', action='store_true')
    parser.add_argument('--Q_inital', type=float, default=1.0, help='Initial Quantization Step')
    parser.add_argument('--initial_Q_w_abs_avg', action='store_true')

    parser.add_argument('--JEPG_alpha', type=float, default=1.0, help='Tempurature scaling')
    parser.add_argument('--JEPG_learning_rate', type=float, default=0.0125, help='Quantization Table Learning Rate')
    parser.add_argument('--alpha_learning_rate', type=float, default=None, help='Alpha Learning Rate')
    parser.add_argument('--block_size', type=int, default=8, help='the experiment id')
    parser.add_argument('--num_bit', type=int, default=8, help='Number of bits to represent DCT coeff')
    parser.add_argument('--min_Q_Step', type=float, default=0.0, choices=range(0, 256), help='Minumum Quantization Step')
    parser.add_argument('--max_Q_Step', type=float, default=255, choices=range(0, 256), help='Maximum Quantization Step')
    parser.add_argument('--num_non_zero_q', type=int, default=5, choices=range(2, 2**11), help='Window size for the reconstruction space')
    parser.add_argument('--hardness', type=float, default=2, help='Hardness of the quantizer')
    parser.add_argument('--std_width_Y', type=float, default=1, help='Width of the standard deviation')
    parser.add_argument('--std_width_CbCr', type=float, default=1, help='Width of the standard deviation')
    parser.add_argument('--jpeg_layers', type=int, default=1, help='Number of JPEG layers')

    
    parser.add_argument('--log_add', type=str, default='', help='add text to the file')
    parser.add_argument('--seed', type=int, default=0, help='seed id, set to 0 if do not want to fix the seed')
    parser.add_argument('--hardness_th', type=float, default=2.5, help='')
    parser.add_argument('--q_max', type=float, default=10, help='maximum q in the initial quantizaiton table')
    parser.add_argument('--q_min', type=float, default=None, help='maximum q in the initial quantizaiton table')
    parser.add_argument("--optimizer", default="", type=str, help="SGD or ADAM")
    parser.add_argument("--JPEG_mode", default="", type=str, help="SGD or ADAM")


    parser.add_argument("--data-path", default="/home/ahamsala/datasets/imagenet/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)