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

from models_cifar100 import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import accuracy, AverageMeter
from dataset.general_dataloader import load_dataset
from helper.loops import validate_attack, validate
from helper.JPEG_layer import *
import cv2
import torchattack
from models import load_model
from PIL import Image
# import matplotlib.pyplot as plt
import pickle


# Define the custom module
class CustomModel(nn.Module):
    def __init__(self, jpeg_layer, underlying_model):
        super(CustomModel, self).__init__()
        self.jpeg_layer = jpeg_layer
        self.underlying_model = underlying_model

    def forward(self, x, *args, **kwargs):
        # Pass input through jpeg_layer
        x = self.jpeg_layer(x)
        # Pass the output of jpeg_layer to the underlying_model
        # along with any additional arguments
        x = self.underlying_model(x, *args, **kwargs)
        return x


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    parser.add_argument('--ADAM_enable', action='store_true')
    parser.add_argument('--JPEG_enable', action='store_true')

    
    parser.add_argument('--JPEG_alpha_trainable', action='store_true')

    parser.add_argument('--zero_dc', action='store_true')
    parser.add_argument('--alpha_scaling', action='store_true')
    parser.add_argument('--alpha_fixed', action='store_true')
    
    parser.add_argument('--initial_Q_w_sensitivity', action='store_true')
    parser.add_argument('--Q_inital', type=float, default=1.0, help='Initial Quantization Step')

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
    parser.add_argument('--model_dir', type=str, default='', help='Mode directory for robustness testing') 


    
    parser.add_argument('--log_add', type=str, default='', help='add text to the file')
    parser.add_argument('--seed', type=int, default=0, help='seed id, set to 0 if do not want to fix the seed')
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
    elif opt.JPEG_alpha_trainable:
        opt.alpha_learning_rate = opt.JEPG_learning_rate
        opt.alpha_setup = "alpha_trainable"
    else:
        opt.alpha_setup = "zero_dc"
        # raise ValueError('You need to setup') 
        

    opt.alpha_setup += "_clampWbits"

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
    
    opt.alpha_setup += "_" + opt.added_layer

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
            opt.model_name = '{}_{}_alpha_lr_{}_alpha_{}_JEPG_lr_{}_numBits_{}_qmax_{}_{}{}_trial_{}'.format(opt.dataset,
                                                                                opt.model, 
                                                                                opt.alpha_learning_rate,
                                                                                opt.JEPG_alpha, 
                                                                                opt.JEPG_learning_rate,
                                                                                opt.num_bit,
                                                                                opt.q_max,
                                                                                opt.alpha_setup,
                                                                                opt.log_add,
                                                                                opt.trial
                                                                                )
    

    print(opt.model_name)

    return opt

attack_data = []

def main():

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        mean=(0.5071, 0.4867, 0.4408)
        std=(0.2675, 0.2565, 0.2761)
        img_shape=(32, 32, 3)
        if opt.JPEG_enable:
            mean_datatloader=(0, 0, 0)
            std_datatloader=(1/255., 1/255., 1/255.)
        else:
            mean_datatloader=mean
            std_datatloader=std

        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, opt=opt, 
                                                            mean=mean_datatloader, std=std_datatloader)
        n_cls = 100
        # model
        underlying_model = model_dict[opt.model](num_classes=n_cls)
    elif opt.dataset == 'CUB200':
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        img_shape=(224, 224, 3)
        
        if opt.JPEG_enable:
            mean_datatloader=(0, 0, 0)
            std_datatloader=(1/255., 1/255., 1/255.)
        else:
            mean_datatloader=mean
            std_datatloader=std
        train_loader, val_loader = load_dataset(name="CUB200", batch_size=opt.batch_size, mean=mean_datatloader, std=std_datatloader)
        n_cls = len(train_loader.dataset.class_names)
        underlying_model = load_model(opt.model, num_classes=n_cls)
    else:
        raise NotImplementedError(opt.dataset)



    if opt.JPEG_enable:
        jpeg_layer = JPEG_layer(opt=opt, img_shape=img_shape, mean=mean, std=std)    
        model_t = CustomModel(jpeg_layer, underlying_model)
    else:
        model_t =  underlying_model
    
    path_t = opt.model_dir

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model_t = model_t.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    check_point = torch.load(path_t)
    model_t.load_state_dict(check_point['model'])

    if opt.JPEG_enable:
        opt = check_point['opt']
        model_t.jpeg_layer.construct(opt)

    criterion_cls = nn.CrossEntropyLoss()


    if torch.cuda.is_available():
        model_t = model_t.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        

    # validate teacher accuracy
    model_t.eval()
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    add_pair(0, teacher_acc.item())
    print('Clean accuracy: ', teacher_acc.item())

    if opt.JPEG_enable:
        lum_qtable =  model_t.jpeg_layer.lum_qtable.view(8,8).clone().detach()
        chrom_qtable =  model_t.jpeg_layer.chrom_qtable.view(8,8).clone().detach()
        print(lum_qtable)

    # Ensure the model is in evaluation mode
    model_t.eval()
    steps = 5
    for eps in np.linspace(1, 4, 10, endpoint=True):
        atk = torchattack.FGSM(model_t, eps=eps/255)
        # alpha = (2.5*eps)/steps
        # atk = torchattack.PGD(model_t, eps=eps/255, alpha=alpha/255, steps=steps, random_start=True)
        if opt.JPEG_enable:
            atk.set_normalization_used(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])
        else:
            atk.set_normalization_used(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        
        atk.set_model_training_mode(model_training=False)
        atk.set_mode_default()
        teacher_acc, _, _ = validate_attack(val_loader, model_t, criterion_cls, opt, atk)
        print('Attacked accuracy for {}/255: '.format(eps), teacher_acc.item())
        
        add_pair(eps, teacher_acc.item())

        if opt.JPEG_enable:
            with torch.no_grad():
                lum_qtable =  model_t.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                chrom_qtable =  model_t.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0)
                alpha_lum =  model_t.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                alpha_chrom =  model_t.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                alpha = torch.cat((alpha_lum, alpha_chrom), 0)
                hardness = alpha * (quantizationTable)**2
                
                print("Hardness --> (Num# of Coeff > {}) Y: {:.3f}, CbCr: {:.3f}".format(opt.hardness_th, (hardness[0]>opt.hardness_th).sum(), (hardness[1]>opt.hardness_th).sum()))
    
                print("Alpha --> Min: {:.2f}, Max: {:.2f}".format(alpha.min().item(), alpha.max().item()))
                print("Quantization Table --> Min: {:.2f}, Max: {:.2f}".format(quantizationTable.min().item(), quantizationTable.max().item()))
        
    dir = "./JPEG_DNN_analysis/Robustness/" + path_t.split("/")[-2] + "_FGSM.npy"
    save_to_pickle(dir, attack_data)


# Function to append a pair of floats to the list
def add_pair(x, y):
    pair = (x, y)  # Create a tuple with the two float numbers
    attack_data.append(pair)  # Append the tuple to the list

def save_to_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)





def validate_image_analysis_without_frontend(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    mean = torch.tensor([0.5071, 0.4867, 0.4408]).cuda()  # Example mean values for each channel
    std = torch.tensor([0.2675, 0.2565, 0.2761]).cuda()   # Example standard deviation values for each channel

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            labels = (target == output.max(-1)[1])
            loss = criterion(output, target)
            print(labels)

            for img_index in range(opt.batch_size//2):
                
                if labels[img_index] == True:
                    flag = "True"
                else:
                    flag = "False"
                

                # breakpoint()
                
                original_image = input.permute(0, 2, 3, 1).cpu().numpy()
                filename = "./JPEG_DNN_analysis/images/{}/original_{}_{}.png".format(flag, img_index + (idx*opt.batch_size//2), labels[img_index])
                image = Image.fromarray(np.uint8(original_image[img_index]))
                image.save(filename)

                # compressed = model[0](input)
                compressed = model(input)
                compressed = denormalize_image(compressed, mean, std).permute(0, 2, 3, 1).cpu().numpy()
                
                psnr = PSNR_cal_RGB(compressed[img_index], original_image[img_index])
                filename = "./JPEG_DNN_analysis/images/{}/compressed_{}_{:.2f}_{}.png".format(flag, img_index + (idx*opt.batch_size//2), psnr, labels[img_index])
                image = Image.fromarray(np.uint8(compressed[img_index]))
                image.save(filename)
                # compressed.min() compressed.max()

            breakpoint()
            
            # if opt.JPEG_enable:
            #     quantizationTable = model[0].quantization_table.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            #     print("Quantization Table --> Min: {:.2f}, Max: {:.2f}".format(quantizationTable.min().item(), quantizationTable.max().item()))
            #     QF_fname = os.path.join(opt.Q_tables_folder, '{experiment}.pt'.format(experiment=opt.model_name))
            #     torch.save(quantization_table, QF_fname)
                
              

            # compressed, input_DCT_block_batch, estimated_reconstructed_space = model[0](input)
            
            # (input_DCT_block_batch[:,:,:,0,0] - estimated_reconstructed_space[:,:,:,0,0]).sum()
            # (input_DCT_block_batch[:,:,:,1,1] - estimated_reconstructed_space[:,:,:,1,1]).sum()
            # (input_DCT_block_batch[:,:,:,7,7] - estimated_reconstructed_space[:,:,:,7,7]).sum()



            # measure accuracy and record loss
        #     acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #     losses.update(loss.item(), input.size(0))
        #     top1.update(acc1[0], input.size(0))
        #     top5.update(acc5[0], input.size(0))

        #     # measure elapsed time
        #     batch_time.update(time.time() - end)
        #     end = time.time()

        #     if idx % opt.print_freq == 0:
        #         print('Test: [{0}/{1}]\t'
        #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #               'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #               'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #                idx, len(val_loader), batch_time=batch_time, loss=losses,
        #                top1=top1, top5=top5))

        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate_image_analysis(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    mean = torch.tensor([0.5071, 0.4867, 0.4408]).cuda()  # Example mean values for each channel
    std = torch.tensor([0.2675, 0.2565, 0.2761]).cuda()   # Example standard deviation values for each channel
    # breakpoint()


    if opt.JPEG_enable:
        quantizationTable = model[0].quantization_table.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        print("Quantization Table --> Min: {:.2f}, Max: {:.2f}".format(quantizationTable.min().item(), quantizationTable.max().item()))
        QF_fname = os.path.join(opt.Q_tables_folder, '{experiment}.pt'.format(experiment=opt.model_name))
        torch.save(quantizationTable, QF_fname)
        print(QF_fname)

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            labels = (target == output.max(-1)[1])
            loss = criterion(output, target)
            # print(labels)

            if not opt.JPEG_enable:
                input = denormalize_image(input, mean, std)

            # compute output
            for img_index in range(opt.batch_size//2):
                
                if labels[img_index] == True:
                    flag = "True"
                else:
                    flag = "False"
                
                # breakpoint()
                
                original_image = input.permute(0, 2, 3, 1).cpu().numpy()
                filename = "./JPEG_DNN_analysis/images/{}/original_{}_{}.png".format(flag, img_index + (idx*opt.batch_size//2), labels[img_index])
                image = Image.fromarray(np.uint8(original_image[img_index]))
                image.save(filename)

                if opt.JPEG_enable:
                    # if input.min() < 0: input += 128
                    compressed = model[0](input)
                    # compressed = model(input)
                    compressed = denormalize_image(compressed, mean, std).permute(0, 2, 3, 1).cpu().numpy()
                    
                    psnr = PSNR_cal_RGB(compressed[img_index], original_image[img_index])
                    filename = "./JPEG_DNN_analysis/images/{}/compressed_{}_{:.2f}_{}.png".format(flag, img_index + (idx*opt.batch_size//2), psnr, labels[img_index])
                    image = Image.fromarray(np.uint8(compressed[img_index]))
                    image.save(filename)
                    # compressed.min() compressed.max()




            # compressed, input_DCT_block_batch, estimated_reconstructed_space = model[0](input)
            
            # (input_DCT_block_batch[:,:,:,0,0] - estimated_reconstructed_space[:,:,:,0,0]).sum()
            # (input_DCT_block_batch[:,:,:,1,1] - estimated_reconstructed_space[:,:,:,1,1]).sum()
            # (input_DCT_block_batch[:,:,:,7,7] - estimated_reconstructed_space[:,:,:,7,7]).sum()



            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

def PSNR_cal_RGB(s, r, max_value=255):
    s = np.array(s, dtype=np.float64) 
    r = np.array(r, dtype=np.float64)
    
    height, width, channel = np.shape(s)
    size = height*width 

    sb,sg,sr = cv2.split(s)
    rb,rg,rr = cv2.split(r)
    
    
    mseb = ((sb-rb)**2).sum()
    mseg = ((sg-rg)**2).sum()
    mser = ((sr-rr)**2).sum()
    mse = (mseb+mseg+mser)
    
    if mse == 0:
        return float('inf')
    
    MSE = mse/(3*size)
    psnr = 10*math.log10(255**2/MSE)
    return round(psnr,2)

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls, train_loader, opt):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    print(model_t)
    model = model_dict[model_t](num_classes=n_cls)
    if opt.JPEG_enable:
        jpeg_layer = JPEG_layer(opt=opt, img_shape=train_loader.dataset.data[0].shape)
        model = nn.Sequential(jpeg_layer, model)
    print(model_path)
    model.load_state_dict(torch.load(model_path)['model'])
    # breakpoint()
    print('==> done')
    return model


def denormalize_image(images, mean, std):
    """
    Denormalizes a minibatch of image tensors using provided mean and standard deviation.
    
    Args:
        images (torch.Tensor): The minibatch of image tensors to denormalize.
                               Shape: (batch_size, channels, height, width)
        mean (torch.Tensor): Tensor of mean values for each channel.
        std (torch.Tensor): Tensor of standard deviation values for each channel.
    
    Returns:
        torch.Tensor: The denormalized minibatch of image tensors.
    """
    # Reshape mean and std tensors to make them compatible with broadcasting
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    
    # Denormalize the entire minibatch of images
    denormalized_images = (images * std + mean)
    denormalized_images =  255* denormalized_images.clamp(min=0,max=1)
    return denormalized_images


if __name__ == '__main__':
    main()
