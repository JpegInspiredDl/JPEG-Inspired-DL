from __future__ import print_function

import os
import time


# import tensorboard_logger.tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import load_model, load_model_imagenet

from dataset.general_dataloader import load_dataset

from helper.util import adjust_learning_rate
from helper.loops import train, validate
from helper.JPEG_layer import *

import numpy as np
import random

import os
import argparse
import socket
# import seaborn as sns
# import matplotlib.pyplot as plt
from dataset.cifar100 import get_cifar100_dataloaders

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,150', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'densenet121', ])
    
    parser.add_argument('--dataset', type=str, default='CUB200', choices=['cifar100', 'CUB200', 'ImageNet'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    parser.add_argument('--JPEG_enable', action='store_true')
    parser.add_argument('--JPEG_alpha_trainable', action='store_true')
    parser.add_argument('--zero_dc', action='store_true')



    parser.add_argument('--JEPG_alpha', type=float, default=1.0, help='Tempurature scaling')
    parser.add_argument('--JEPG_learning_rate', type=float, default=0.0125, help='Quantization Table Learning Rate')
    parser.add_argument('--alpha_learning_rate', type=float, default=None, help='Alpha Learning Rate')
    parser.add_argument('--Q_inital', type=float, default=1.0, help='Initial Quantization Step')
    parser.add_argument('--block_size', type=int, default=8, help='the experiment id')
    parser.add_argument('--num_bit', type=int, default=11, help='Number of bits to represent DCT coeff')
    parser.add_argument('--min_Q_Step', type=float, default=0.0, choices=range(0, 256), help='Minumum Quantization Step')
    parser.add_argument('--max_Q_Step', type=float, default=None, choices=range(1, 256), help='Maximum Quantization Step')
    parser.add_argument('--num_non_zero_q', type=int, default=5, choices=range(2, 2**11), help='Window size for the reconstruction space')
    parser.add_argument('--hardness', type=float, default=2, help='Hardness of the quantizer')
    parser.add_argument('--std_width_Y', type=float, default=1, help='Width of the standard deviation')
    parser.add_argument('--std_width_CbCr', type=float, default=1, help='Width of the standard deviation')
    parser.add_argument('--jpeg_layers', type=int, default=1, help='Number of JPEG layers')

    
    
    parser.add_argument('--std_width', type=int, default=2, choices=range(2, 10), help='Width of the standard deviation')
    parser.add_argument('--p_quantile', type=float, default=0.2, help='Width of the standard deviation')


    parser.add_argument('--log_file', type=str, default='_alpha_untrainable', help='add text to the file', choices=['', '_alpha_per_q', '_alpha_untrainable'])
    parser.add_argument('--seed', type=int, default=0, help='seed id, set to 0 if do not want to fix the seed')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of micro-batches to accumulate')

    

    
    opt = parser.parse_args()
    

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        # opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/{}/models'.format(opt.dataset)
        # opt.tb_path = './save/tensorboard'
        opt.log_pth = './save/{}/teacher_log/'.format(opt.dataset)
        opt.Q_tables_pth = './save/{}/teacher_Q_tables/'.format(opt.dataset)
        opt.alpha_pth = './save/{}/teacher_Alpha/'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    if opt.JPEG_alpha_trainable:
        opt.log_file = "_alpha_per_q"
    else:
        opt.log_file = "_alpha_untrainable_2_tables"

    if opt.JPEG_enable:
        opt.added_layer = "JPEG"
        # opt.model_name = '{}_{}_alpha_lr_{}_alpha_{}_JEPG_lr_{}_num_non_zero_q_{}_numBits_{}_Q_initial_{}_Q_min_{}{}_trial_{}'.format(opt.dataset,
        #                                                                         opt.model, 
        #                                                                         opt.alpha_learning_rate,
        #                                                                         opt.JEPG_alpha, 
        #                                                                         opt.JEPG_learning_rate,
        #                                                                         opt.num_non_zero_q,
        #                                                                         opt.num_bit,
        #                                                                         opt.Q_inital,
        #                                                                         opt.min_Q_Step,
        #                                                                         opt.log_file,
        #                                                                         opt.trial
        #                                                                         )
        opt.model_name = '{}_{}_alpha_lr_{}_JEPG_lr_{}_numBits_{}_hardness_{}_Q_min_{}{}_trial_{}'.format(opt.dataset,
                                                                                opt.model, 
                                                                                opt.alpha_learning_rate,
                                                                                opt.JEPG_learning_rate,
                                                                                opt.num_bit,
                                                                                opt.hardness,
                                                                                opt.min_Q_Step,
                                                                                opt.log_file,
                                                                                opt.trial
                                                                                )
    else:
        opt.added_layer = "vanilla"
        opt.model_name = '{}_{}_trial_{}'.format(opt.dataset, opt.model,  opt.trial)

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)
    opt.dir_initial = "./Initialization_alpha_Q_tables/{}_std.pt".format(opt.dataset)
    print(opt.model_name)
    print("Reading Q and Alpha from: ", opt.dir_initial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_key = 'T:{}_lr_{}_decay_{}_log_{}'.format(opt.model, opt.learning_rate,
                                                         opt.weight_decay, opt.added_layer)

    opt.log_folder = os.path.join(opt.log_pth, opt.log_key)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    opt.alpha_folder = os.path.join(opt.alpha_pth, opt.log_key)
    if not os.path.isdir(opt.alpha_folder):
        os.makedirs(opt.alpha_folder)

    opt.Q_tables_folder = os.path.join(opt.Q_tables_pth, opt.log_key)
    if not os.path.isdir(opt.Q_tables_folder):
        os.makedirs(opt.Q_tables_folder)


    return opt

def load_jpeg_model(model_path, mean, std, n_cls, opt, img_shape):
    underlying_model = load_model(opt.model, num_classes=n_cls, pretrained=False)
    checkpoint = torch.load(model_path)
    if opt.JPEG_enable:
        jpeg_layer = JPEG_layer(opt=opt, img_shape=img_shape, mean=mean, std=std)        
        model = CustomModel(jpeg_layer, underlying_model)
        model.load_state_dict(checkpoint['model'])
        opt = checkpoint['opt']
        model.jpeg_layer.construct(opt)
    else:
        model =  underlying_model
        model.load_state_dict(checkpoint['model'])
    return model, opt

def check_val():
    opt = parse_option()
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    if opt.JPEG_enable:
        mean_datatloader=(0, 0, 0)
        std_datatloader=(1/255., 1/255., 1/255.)
    else:
        mean_datatloader=mean
        std_datatloader=std


    if opt.dataset == 'CUB200':
        batch_size=int(opt.batch_size/opt.accumulation_steps)
        train_loader, val_loader = load_dataset(name="CUB200", mean=mean_datatloader, std=std_datatloader, batch_size=batch_size)
        n_cls = len(train_loader.dataset.class_names)
        # model
        # validate teacher accuracy
        # model_path = "./save/CUB200/pretrained/resnet18/resnet18_best.pth" 
        model_path = "./save/CUB200/models/CUB200_resnet18_alpha_lr_1.0_alpha_15.0_JEPG_lr_0.3_num_non_zero_q_5_numBits_11_Q_initial_1.0_Q_min_1.0_alpha_per_q_trial_1/resnet18_best.pth"
        
        model, opt = load_jpeg_model(model_path, mean, std, n_cls, opt, img_shape=(224, 224, 3))
    elif opt.dataset == 'ImageNet':
        batch_size=int(opt.batch_size/opt.accumulation_steps)
        train_loader, val_loader = load_dataset(name="ImageNet", root='/home/l44ye/datasets', mean=mean, std=std, batch_size=batch_size)
        # n_cls = len(train_loader.dataset.class_names)
        model = load_model_imagenet(opt.model, pretrained=True)

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    teacher_acc, _, _ = validate(val_loader, model, criterion, opt)
    print('Model accuracy: ', teacher_acc.item())



def main():
    best_acc = 0
    opt = parse_option()

    if opt.seed:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    

    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    if opt.JPEG_enable:
        mean_datatloader=(0, 0, 0)
        std_datatloader=(1/255., 1/255., 1/255.)
    else:
        mean_datatloader=mean
        std_datatloader=std

    if opt.dataset == 'cifar100':
        NUM_BLOCKs = 16
        img_shape=(32, 32, 3)
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, opt=opt, mean=mean_datatloader, std=std_datatloader)
        n_cls = 100
    elif opt.dataset == 'CUB200':
        NUM_BLOCKs = 784
        img_shape=(224, 224, 3)
        batch_size=int(opt.batch_size/opt.accumulation_steps)
        train_loader, val_loader = load_dataset(name="CUB200", mean=mean_datatloader, std=std_datatloader, batch_size=batch_size)
        n_cls = len(train_loader.dataset.class_names)
        underlying_model = load_model(opt.model, num_classes=n_cls, pretrained=False)
    elif opt.dataset == 'ImageNet':
        NUM_BLOCKs = 784
        img_shape=(224, 224, 3)
        batch_size=int(opt.batch_size/opt.accumulation_steps)
        train_loader, val_loader = load_dataset(name="ImageNet", root='/home/ahamsala/datasets/imagenet', mean=mean_datatloader, std=std_datatloader, batch_size=batch_size)
        # n_cls = len(train_loader.dataset.class_names)
        # model = load_model_imagenet(opt.model, pretrained=True)
        # underlying_model = load_model(opt.model, pretrained=False)
    else:
        raise NotImplementedError(opt.dataset)

    # model

    if opt.JPEG_enable:      
        opt.dir_initial_std = "./Initialization_alpha_Q_tables/{}_std.pt".format(opt.dataset) 
        opt.dir_initial_mean = "./Initialization_alpha_Q_tables/{}_mean.pt".format(opt.dataset) 
        opt.dir_initial_abs_avg = "./Initialization_alpha_Q_tables/{}_abs_avg.pt".format(opt.dataset) 
        
        jpeg_layer = JPEG_layer(opt=opt, img_shape=img_shape, mean=mean, std=std)
        # model = CustomModel(jpeg_layer, underlying_model)

    else:
        model =  underlying_model

 
    criterion = nn.CrossEntropyLoss()


    if torch.cuda.is_available():
        jpeg_layer= jpeg_layer.cuda()
        criterion = criterion.cuda()
        if not opt.seed:
            cudnn.benchmark = True

    std_data_Y = torch.zeros((8, 8),dtype=torch.float)
    std_data_CbCr = torch.zeros((8, 8),dtype=torch.float)
    mean_data_Y  = torch.zeros((8, 8),dtype=torch.float)
    mean_data_CbCr  = torch.zeros((8, 8),dtype=torch.float)

    # tables = torch.load("./Initialization_alpha_Q_tables/{}_mean.pt".format(opt.dataset))
    # mean_data_Y = tables["mean_Y"]
    # mean_data_CbCr = tables["mean_CbCr"]
    if torch.cuda.is_available():
        std_data_Y = std_data_Y.cuda()
        std_data_CbCr = std_data_CbCr.cuda()
        mean_data_Y = mean_data_Y.cuda()
        mean_data_CbCr = mean_data_CbCr.cuda()



    
    count = 0
    
    for idx, (input, target) in enumerate(train_loader):
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        _, input_DCT_block_batch, _ = jpeg_layer(input, analysis=True)        
        mean_data_Y += torch.abs(input_DCT_block_batch[:, 0, ...].squeeze(-1)).sum(dim=(0,1))
        mean_data_CbCr += torch.abs(input_DCT_block_batch[:, 1:3, ...].squeeze(-1)).sum(dim=(0,1,2))

        # std_data_Y += (input_DCT_block_batch[:, 0, ...].squeeze(-1) - mean_data_Y.unsqueeze(0).unsqueeze(1)).pow(2).sum(dim=(0,1))
        # std_data_CbCr += (input_DCT_block_batch[:, 1:3, ...].squeeze(-1) - mean_data_CbCr.unsqueeze(0).unsqueeze(1).unsqueeze(2)).pow(2).sum(dim=(0,1,2))
        count += input.size(0)
        # break
    
    print(count)

    
    tables = {}
    tables["abs_avg_Y"] = mean_data_Y / (count*NUM_BLOCKs)
    tables["abs_avg_CbCr"] = mean_data_CbCr / (count*2*NUM_BLOCKs)
    torch.save(tables, "./Initialization_alpha_Q_tables/{}_abs_avg.pt".format(opt.dataset))

    # tables = {}
    # tables["std_Y"] = torch.sqrt(std_data_Y / (count*NUM_BLOCKs))
    # tables["std_CbCr"] = torch.sqrt(std_data_CbCr / (count*2*NUM_BLOCKs))
    # torch.save(tables, "./Initialization_alpha_Q_tables/{}_std.pt".format(opt.dataset))
    # std_data_Y = tables["std_Y"].cpu().clone()
    # std_data_CbCr = tables["std_CbCr"].cpu().clone()

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(tables["abs_sum_Y"], annot=True, xticklabels=range(8), yticklabels=range(8), cmap="YlGnBu")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Standard Deviation Y channel')
    # plt.savefig("./Initialization_alpha_Q_tables/{}_abs_sum_Y_Channel.png".format(opt.dataset), dpi=600)
    # plt.close()


    # plt.figure(figsize=(10, 8))
    # sns.heatmap(tables["abs_sum_CbCr"] , annot=True, xticklabels=range(8), yticklabels=range(8), cmap="YlGnBu")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Standard Deviation CbCr channel')
    # plt.savefig("./Initialization_alpha_Q_tables/{}_abs_sum_CbCr_Channel.png".format(opt.dataset), dpi=600)
    # plt.close()

if __name__ == '__main__':
    main()
    # check_val()
    # profiling()
