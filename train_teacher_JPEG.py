from __future__ import print_function

import os
import argparse
import socket
import time


# import tensorboard_logger.tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import load_model, load_model_imagenet

from dataset.general_dataloader import load_dataset


from helper.util import adjust_learning_rate_JPEG, accuracy, AverageMeter
from helper.loops import train_JPEG, validate
from helper.JPEG_layer import *

import numpy as np
import random

import os
import argparse
import socket


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
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet18', 'densenet121', 'Mnasnet',
                                 'MobileNet_V2', 'ShuffleV1', 'ShuffleV2', 'ResNet50'])
    parser.add_argument('--dataset', type=str, default='CUB200', choices=['CUB200', "STANFORD120", "Food101", "Flowers", 'ImageNet'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    parser.add_argument('--ADAM_enable', action='store_true')
    parser.add_argument('--JPEG_enable', action='store_true')

    parser.add_argument('--plot_img', action='store_true')

    
    parser.add_argument('--JPEG_alpha_trainable', action='store_true')

    parser.add_argument('--zero_dc', action='store_true')
    parser.add_argument('--alpha_scaling', action='store_true')
    parser.add_argument('--alpha_fixed', action='store_true')
    
    parser.add_argument('--initial_Q_w_sensitivity', action='store_true')
    parser.add_argument('--initial_Q_abs_avg', action='store_true')

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
    parser.add_argument('--seed', type=int, default=0, help='seed id, set to 0 if do not want to fix the seed')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of micro-batches to accumulate')
    parser.add_argument('--hardness_th', type=float, default=2.5, help='')
    parser.add_argument('--q_max', type=float, default=10, help='maximum q in the initial quantizaiton table')
    

    opt = parser.parse_args()
    
    # # This will be only for KD, NOT CE benchmarking.
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

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


    if opt.zero_dc:
        opt.alpha_setup = "zero_dc"
    elif opt.alpha_scaling:
        opt.alpha_setup = "alpha_scaling"
    elif opt.alpha_fixed:
        opt.alpha_setup = "alpha_fixed"
    else:
        raise ValueError('You need to setup') 

    opt.alpha_setup += "_clampWbits"

    if opt.initial_Q_abs_avg:
        opt.alpha_setup += "_Init_Q_abs_avg"
    elif opt.initial_Q_w_sensitivity:
        opt.alpha_setup += "_initial_Q_w_sensitivity"
    else:
        opt.alpha_setup += "_Init_Q_Ones"

    if opt.ADAM_enable:
        opt.adam = "_ADAM_"
    else:
        opt.adam = ""

    if opt.max_Q_Step == 0:
        opt.max_Q_Step = None

    opt.dir_initial_std = "./Initialization_alpha_Q_tables/{}_std.pt".format(opt.dataset) 
    opt.dir_initial_mean = "./Initialization_alpha_Q_tables/{}_mean.pt".format(opt.dataset) 
    opt.dir_initial_abs_avg = "./Initialization_alpha_Q_tables/{}_abs_avg.pt".format(opt.dataset) 
    opt.sensitvity_dir = "./Senstivity/Senstivity_{}/SenMap/".format(opt.dataset)
    
    if opt.ADAM_enable:
        opt.added_layer = "JPEG_ADAMs"
    else:
        opt.added_layer = "JPEG_SGD"

    if opt.jpeg_layers > 1:
        opt.model_name = '{}_{}_alpha_lr_{}_alpha_{}_JEPG_lr_{}_hardness_{}_Q_min_{}_std_Y_{}_std_CbCr_{}_num_layer_{}_{}{}_trial_{}'.format(opt.dataset,
                                                                                opt.model, 
                                                                                opt.alpha_learning_rate,
                                                                                opt.JEPG_alpha, 
                                                                                opt.JEPG_learning_rate,
                                                                                opt.hardness,
                                                                                opt.min_Q_Step,
                                                                                opt.std_width_Y,
                                                                                opt.std_width_CbCr,
                                                                                opt.jpeg_layers,
                                                                                opt.alpha_setup,
                                                                                opt.log_add,
                                                                                opt.trial
                                                                                )        
    else:
        # opt.model_name = '{}_{}_JEPG_lr_{}_hardness_{}_Q_min_{}_Q_max_{}_std_Y_{}_std_CbCr_{}_{}{}_trial_{}'.format(opt.dataset,
        #                                                                         opt.model, 
        #                                                                         opt.JEPG_learning_rate,
        #                                                                         opt.hardness,
        #                                                                         opt.min_Q_Step,
        #                                                                         opt.max_Q_Step,
        #                                                                         opt.std_width_Y,
        #                                                                         opt.std_width_CbCr,
        #                                                                         opt.alpha_setup,
        #                                                                         opt.log_add,
        #                                                                         opt.trial
        #                                                                         )

        # opt.model_name = '{}_{}_alpha_lr_{}_alpha_{}_JEPG_lr_{}_hardness_{}_numBits_{}_Q_min_{}_Q_max_{}_{}{}_trial_{}'.format(opt.dataset,
        #                                                                         opt.model, 
        #                                                                         opt.alpha_learning_rate,
        #                                                                         opt.JEPG_alpha, 
        #                                                                         opt.JEPG_learning_rate,
        #                                                                         opt.hardness,
        #                                                                         opt.num_bit,
        #                                                                         opt.min_Q_Step,
        #                                                                         opt.max_Q_Step,
        #                                                                         opt.alpha_setup,
        #                                                                         opt.log_add,
        #                                                                         opt.trial
        #                                                                         )
            opt.model_name = '{}_{}_alpha_lr_{}_alpha_{}_JEPG_lr_{}_numBits_{}_qmax_{}_{}{}{}_trial_{}'.format(opt.dataset,
                                                                                opt.model, 
                                                                                opt.alpha_learning_rate,
                                                                                opt.JEPG_alpha, 
                                                                                opt.JEPG_learning_rate,
                                                                                opt.num_bit,
                                                                                opt.q_max,
                                                                                opt.alpha_setup,
                                                                                opt.log_add,
                                                                                opt.adam,
                                                                                opt.trial
                                                                                )
    

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)

    print(opt.model_name)
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
    jpeg_layer = JPEG_layer(opt=opt, img_shape=img_shape, mean=mean, std=std)        
    model = CustomModel(jpeg_layer, underlying_model)
    model.load_state_dict(checkpoint['model'])
    opt = checkpoint['opt']
    model.jpeg_layer.construct(opt)
    return model, opt



def main():
    best_acc = 0

    opt = parse_option()

    if torch.cuda.is_available():
        cudnn.benchmark = True
    
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
    mean_datatloader=(0, 0, 0)
    std_datatloader=(1/255., 1/255., 1/255.)

    if opt.dataset in ['CUB200', 'STANFORD120', 'Flowers', 'Food101']:
        batch_size=int(opt.batch_size/opt.accumulation_steps)
        train_loader, val_loader = load_dataset(name=opt.dataset, mean=mean_datatloader, std=std_datatloader, batch_size=batch_size)
        n_cls = train_loader.dataset.num_classes
        if opt.model in ['resnet18', 'densenet121']:
            underlying_model = load_model(opt.model, num_classes=n_cls, pretrained=False)
        else:
            underlying_model = load_model_imagenet(opt.model, pretrained=False)
            underlying_model.classifier[-1] = nn.Linear(underlying_model.classifier[-1].in_features, n_cls)

    elif opt.dataset == 'ImageNet':
        batch_size=int(opt.batch_size/opt.accumulation_steps)
        train_loader, val_loader = load_dataset(name="ImageNet", root='/home/l44ye/datasets', mean=mean_datatloader, std=std_datatloader, batch_size=batch_size)
        # n_cls = len(train_loader.dataset.class_names)
        model = load_model_imagenet(opt.model, pretrained=True)
        # underlying_model = load_model(opt.model, pretrained=False)
    else:
        raise NotImplementedError(opt.dataset)


    # --------------------------

    # opt.Nexample = 5000
    # pretrained_underlying_model = underlying_model

    # # jpeg_layer = JPEG_layer(opt=opt, img_shape=img_shape, mean=mean, std=std)        
    # # model = CustomModel(jpeg_layer, underlying_model)
    # # model.load_state_dict(torch.load(model_path)['model'])
    # # model_path = "./save/models/{}/{}_last.pth".format(opt.model, opt.model)



    # model_path = "./save/CUB200/pretrained/{}_vanilla/{}_best.pth".format(opt.model, opt.model)
    # pretrained_underlying_model.load_state_dict(torch.load(model_path)['model'])
    # Y_sens, Cb_sens, Cr_sens = get_sens(train_loader, pretrained_underlying_model, opt, mean, std, save=True)
    # exit(0)


    # -------------------------

    jpeg_layer = JPEG_layer(opt=opt, img_shape=(224, 224, 3), mean=mean, std=std)
    
    model = CustomModel(jpeg_layer, underlying_model)
    
    optimizer_underlying_model_data = [
                        {'params': model.underlying_model.parameters(), 'lr': opt.learning_rate, 'momentum': opt.momentum, 'weight_decay': opt.weight_decay},
                    ]

    if  opt.ADAM_enable:
        optimizer_JPEG_layer_data = [
                            {'params': model.jpeg_layer.lum_qtable, 'lr': opt.JEPG_learning_rate,},
                            {'params': model.jpeg_layer.chrom_qtable, 'lr': opt.JEPG_learning_rate,},
                            ]
    else:
        optimizer_JPEG_layer_data = [
                            {'params': model.jpeg_layer.lum_qtable, 'lr': opt.JEPG_learning_rate, 'momentum': opt.momentum,},
                            # {'params': model.jpeg_layer.chrom_qtable, 'lr': (opt.JEPG_learning_rate)/np.sqrt(2), 'momentum': opt.momentum,},
                            {'params': model.jpeg_layer.chrom_qtable, 'lr': opt.JEPG_learning_rate, 'momentum': opt.momentum,},
                            ]
    
    if opt.JPEG_alpha_trainable:
        if  opt.ADAM_enable:
            optimizer_JPEG_layer_data.append({'params': model.jpeg_layer.alpha_lum, 'lr': opt.alpha_learning_rate,})
            optimizer_JPEG_layer_data.append({'params': model.jpeg_layer.alpha_chrom, 'lr': opt.alpha_learning_rate,})
        else:
            optimizer_JPEG_layer_data.append({'params': model.jpeg_layer.alpha_lum, 'lr': opt.alpha_learning_rate, 'momentum': opt.momentum,})
            optimizer_JPEG_layer_data.append({'params': model.jpeg_layer.alpha_chrom, 'lr': opt.alpha_learning_rate, 'momentum': opt.momentum,})

    
    optimizer_underlying_model = optim.SGD(optimizer_underlying_model_data)
    
    if opt.ADAM_enable:
        optimizer_JPEG_Layer = optim.Adam(optimizer_JPEG_layer_data)
    else:
        optimizer_JPEG_Layer = optim.SGD(optimizer_JPEG_layer_data)
    
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model= model.cuda()
        criterion = criterion.cuda()
        if not opt.seed:
            cudnn.benchmark = True





    # log file
    log_fname = os.path.join(opt.log_folder, '{experiment}.txt'.format(experiment=opt.model_name))
    print(log_fname)

    str_q_alpha = ""
    # routine
    time1_total = time.time()
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate_JPEG(epoch, opt, optimizer_underlying_model, optimizer_JPEG_Layer)
        print("==> training...")

        lum_qtable =  model.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        chrom_qtable =  model.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0)
        alpha_lum =  model.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        alpha_chrom =  model.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        alpha = torch.cat((alpha_lum, alpha_chrom), 0)
        hardness = alpha * (quantizationTable)**2
        
        print("Hardness --> (Num# of Coeff > {}) Y: {:.3f}, CbCr: {:.3f}".format(opt.hardness_th, (hardness[0]>opt.hardness_th).sum(), (hardness[1]>opt.hardness_th).sum()))
        print("Hardness --> Min: {:.3f}, Max: {:.3f}".format(hardness.min().item(), hardness.max().item()))
        print("Q Table  --> Min: {:.3f}, Max: {:.2f}".format(quantizationTable.min().item(), quantizationTable.max().item()))
        print("Alpha    --> Min: {:.3f}, Max: {:.3f}".format(alpha.min().item(), alpha.max().item()))
        # print("")
        # print("Hardness_lum\n" , hardness[0])
        # print("Hardness_chrom\n" , hardness[1])
        # print("")
        str_q_alpha = "| {:.3f} \t {:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t".format(
                                                                        opt.hardness_th,
                                                                        (hardness[0]>opt.hardness_th).sum(), (hardness[1]>opt.hardness_th).sum(),
                                                                        hardness.min().item(), hardness.max().item(),
                                                                        quantizationTable.min().item(), quantizationTable.max().item(),
                                                                        alpha.min().item(), alpha.max().item(),
                                                                        )

            
        if opt.JPEG_alpha_trainable:
            alpha_lum =  model.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alpha_chrom =  model.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alpha = torch.cat((alpha_lum, alpha_chrom), 0)
            print("Alpha --> Min: {:.5f}, Max: {:.5f}".format(alpha.min().item(), alpha.max().item()))
            str_q_alpha += "\t" +  str(alpha.min().item())  + "\t" + str(alpha.max().item())  + "\t" 

                
        time1 = time.time()
        train_acc, train_loss = train_JPEG(epoch, train_loader, model, criterion, optimizer_underlying_model, optimizer_JPEG_Layer, opt)
        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        
        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc

            QF_fname = os.path.join(opt.Q_tables_folder, '{experiment}_best.pt'.format(experiment=opt.model_name))
            alpha_fname = os.path.join(opt.alpha_folder, '{experiment}_best.pt'.format(experiment=opt.model_name))
            lum_qtable =  model.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            chrom_qtable =  model.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0).cpu()
            torch.save(quantizationTable, QF_fname)

            alpha_lum =  model.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alpha_chrom =  model.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alpha = torch.cat((alpha_lum, alpha_chrom), 0).cpu()
            torch.save(alpha, alpha_fname)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'best_acc': best_acc,
                'opt': opt,
                'optimizer_JPEG_layer': optimizer_JPEG_Layer.state_dict(),
                'optimizer_underlying_model': optimizer_underlying_model.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        with open(log_fname, 'a') as log:
            line = str(epoch) + "\t" + \
                   str(test_acc.cpu().numpy()) + "\t" +  \
                   str(best_acc.item()) + "\t" +  \
                   str(test_loss)   + "\t" + \
                   str(train_loss)  + "\t" + \
                   str_q_alpha + "\t" + \
                    '\n'
            log.write(line)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'opt': opt,
                'optimizer_JPEG_layer': optimizer_JPEG_Layer.state_dict(),
                'optimizer_underlying_model': optimizer_underlying_model.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            # torch.save(state, save_file)
        
        if opt.seed and epoch >= 1: break
        

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc.item())

    
    exp_txt = open('./results/./results/./results/{}_{}_experimetal_Results_JPEG_DL.txt'.format(opt.dataset, opt.model), 'a+')
    exp_txt.write(opt.model_name +"\t"+ str(epoch) + "\t" + str(best_acc.item()) + "\t" + str((time.time() - time1_total)/ 3600) + " hr" + "\n") # Write some text
    exp_txt.close() # Close the file

    # save model
    state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'best_acc': best_acc,
            'accuracy': test_acc,
            'opt': opt,
            'optimizer_JPEG_layer': optimizer_JPEG_Layer.state_dict(),
            'optimizer_underlying_model': optimizer_underlying_model.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)

def profiling():
    import cProfile
    import pstats
 
    with cProfile.Profile() as pr:
        main()
 
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename='profile_vanilla.prof')

if __name__ == '__main__':
    main()
    # check_val()
    # profiling()
