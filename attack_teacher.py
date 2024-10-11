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
import matplotlib.pyplot as plt


from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
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
                        choices=['resnet18', 'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'CUB200'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    parser.add_argument('--JPEG_enable', action='store_true')
    parser.add_argument('--JEPG_alpha', type=float, default=10, help='Tempurature scaling')
    parser.add_argument('--JPEG_alpha_trainable', action='store_true')
    parser.add_argument('--JEPG_learning_rate', type=float, default=0.0001, help='Quantization Table Learning Rate')
    parser.add_argument('--Q_inital', type=float, default=10, help='Initial Quantization Step')
    parser.add_argument('--block_size', type=int, default=8, help='the experiment id')
    parser.add_argument('--num_bit', type=int, default=8, help='Number of bits to represent DCT coeff')
    parser.add_argument('--min_Q_Step', type=float, default=1, help='Minumum Quantization Step')
    parser.add_argument('--max_Q_Step', type=float, default=255, help='Maximum Quantization Step')
    parser.add_argument('--log_file', type=str, default='', help='add text to the file')
    
    

    

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'
        opt.log_pth = './save/teacher_log/'
        opt.Q_tables_pth = './save/teacher_Q_tables/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    if opt.JPEG_enable:
        opt.added_layer = "JPEG"
        # opt.model_name = '{}_JEPG_alpha_{}_JEPG_lr_{}_Q_inital_{}_block_size_{}_num_bit_{}_trial_{}'.format(opt.model, opt.JEPG_alpha, 
        #                                                                         opt.JEPG_learning_rate,
        #                                                                         opt.Q_inital,
        #                                                                         opt.block_size,
        #                                                                         opt.num_bit,
        #                                                                         opt.trial)
        opt.model_name = '{}_JEPG_alpha_{}_JEPG_lr_{}_Q_inital_{}_Min_Q_{}_block_size_{}_num_bit_{}_trial_{}{}'.format(opt.model, opt.JEPG_alpha, 
                                                                                opt.JEPG_learning_rate,
                                                                                opt.Q_inital,
                                                                                opt.min_Q_Step,
                                                                                opt.block_size,
                                                                                opt.num_bit,
                                                                                opt.trial,
                                                                                opt.log_file
                                                                                )
    else:
        opt.added_layer = "vanilla"
        opt.model_name = '{}_trial_{}'.format(opt.model,  opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_key = 'T:{}_lr_{}_decay_{}_log_{}'.format(opt.model, opt.learning_rate,
                                                         opt.weight_decay,opt.added_layer)

    opt.log_folder = os.path.join(opt.log_pth, opt.log_key)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    opt.Q_tables_folder = os.path.join(opt.Q_tables_pth, opt.log_key)
    if not os.path.isdir(opt.Q_tables_folder):
        os.makedirs(opt.Q_tables_folder)

    return opt




def main():
    best_acc = 0
    opt = parse_option()

    # model
    # vgg8_cifar100_lr_0.05_decay_0.0005_log_vanilla_trial_1	240	70.83
    # opt.path_t = "./save_v1/models_org/vgg8_cifar100_lr_0.05_decay_0.0005_log_vanilla_trial_2/vgg8_last.pth"
    # opt.JPEG_enable = False


    # vgg8_JEPG_alpha_0.5_JEPG_lr_0.4_Q_inital_1.0_Min_Q_1.0_block_size_8_num_bit_8_trial_2	240	71.3499984741211
    # opt.path_t = "./save/models/vgg8_JEPG_alpha_0.5_JEPG_lr_0.4_Q_inital_1.0_Min_Q_1.0_block_size_8_num_bit_8_trial_2/vgg8_last.pth"
    # opt.JEPG_alpha = 0.5

    # opt.path_t = "./save/models/vgg8_JEPG_alpha_0.5_JEPG_lr_0.15_Q_inital_1.0_Min_Q_1.0_block_size_8_num_bit_8_trial_2/vgg8_last.pth"
    # opt.JEPG_alpha = 0.5

    # vgg8_JEPG_alpha_20.0_JEPG_lr_0.025_Q_inital_1.0_Min_Q_1.0_block_size_8_num_bit_8_trial_2	240	71.23999786376953
    # opt.path_t = "./save/models/vgg8_JEPG_alpha_20.0_JEPG_lr_0.025_Q_inital_1.0_Min_Q_1.0_block_size_8_num_bit_8_trial_2/vgg8_best.pth"
    

    # vgg8_JEPG_alpha_1.0_JEPG_lr_0.2_Q_inital_1.0_Min_Q_1.0_block_size_8_num_bit_8_trial_2	240	70.91999816894531
    # opt.path_t = "./save/models/vgg8_JEPG_alpha_1.0_JEPG_lr_0.2_Q_inital_1.0_Min_Q_1.0_block_size_8_num_bit_8_trial_2/vgg8_best.pth"
    # opt.JEPG_alpha = 1

    
    
    # vgg8_JEPG_alpha_10.0_JEPG_lr_0.05_Q_inital_1.0_Min_Q_0.0_alpha_per_q_trial_1	240	70.81999969482422
    # opt.path_t = "./save/models/vgg8_JEPG_alpha_10.0_JEPG_lr_0.05_Q_inital_1.0_Min_Q_0.0_alpha_per_q_trial_1/vgg8_best.pth"
    # opt.JEPG_alpha = 10



    # opt.path_t = "./save/CUB200/models/CUB200_resnet18_alpha_lr_0.3_alpha_0.7_JEPG_lr_0.3_alpha_untrainable_trial_1/resnet18_best.pth"
    # opt.JEPG_alpha = 0.7


    # opt.path_t =  "./save/CUB200/models/CUB200_resnet18_alpha_lr_None_alpha_0.7_JEPG_lr_0.3_numbits_7_Q_initial_1.0_Q_min_1.0_alpha_untrainable_trial_2/resnet18_best.pth"
    # opt.JEPG_alpha = 0.7

    # opt.path_t = "./save/CUB200/pretrained/resnet18/resnet18_best.pth"
    # opt.JPEG_enable = False

    # CUB200_resnet18_alpha_lr_0.3_alpha_15.0_JEPG_lr_0.3_numBits_8_Q_initial_1.0_Q_min_1.0_alpha_untrainable_trial_3	200	56.610286712646484


    opt.path_t =  "./save/CUB200/models/CUB200_resnet18_alpha_lr_0.3_alpha_15.0_JEPG_lr_0.3_numBits_8_Q_initial_1.0_Q_min_1.0_alpha_untrainable_trial_3/resnet18_best.pth"
    opt.JEPG_alpha = 15
    opt.num_bit = 8
    opt.batch_size = 4

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

        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, opt=opt)
        n_cls = 100
        # model
        underlying_model = model_dict[opt.model](num_classes=n_cls, return_input=True)
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
        underlying_model = load_model(opt.model, num_classes=n_cls, pretrained=False)
    else:
        raise NotImplementedError(opt.dataset)

    

    if opt.JPEG_enable:
        jpeg_layer = JPEG_layer(opt=opt, img_shape=img_shape, mean=mean, std=std)    
        model_t = CustomModel(jpeg_layer, underlying_model)
        # model_t = nn.Sequential(model_t.jpeg_layer, model_t.underlying_model)
        # model_t = nn.Sequential(jpeg_layer, underlying_model)
        # model_t = nn.Sequential(jpeg_layer)
    else:
        model_t =  underlying_model

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model_t = model_t.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True




    if opt.dataset == 'cifar100':
        model_t = load_teacher(opt.path_t, n_cls, train_loader, opt) 
    elif opt.dataset == 'CUB200':
        model_t.load_state_dict(torch.load(opt.path_t)['model'])

    if opt.JPEG_enable:
        model_t = nn.Sequential(model_t.jpeg_layer, model_t.underlying_model)

    criterion_cls = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model_t = model_t.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    
    # validate teacher accuracy
    model_t.eval()

    acc = []

    # teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    # acc.append(teacher_acc.item())
    # print('Clean accuracy: ', teacher_acc.item())

    # teacher_acc, _, _ = validate_image_analysis(val_loader, model_t, criterion_cls, opt)

    '''
    compressed_113_16.98_True
    compressed_239_18.88_True
    compressed_238_21.00_True
    compressed_215_21.31_True
    '''
    
    # image_path = "./JPEG_DNN_analysis/images/True/original_113_True.png"

    # image_path = "./JPEG_DNN_analysis/images/False/original_84_False.png"
    # output_dir = "./JPEG_DNN_analysis/images/"


    image_path = "./JPEG_DNN_analysis/images_8_bits/True/original_84_True.png"
    output_dir = "./JPEG_DNN_analysis/images_8_bits/"


    image = Image.open(image_path)

    preprocess = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Dictionary to store the feature maps
    feature_maps = {}

    # Define a hook function
    def hook_fn(module, input, output):
        feature_maps[module] = output

    
    if opt.JPEG_enable:
        # Register hooks to the desired layers (e.g., first conv layer, and the layers of each block)
        model_t[1].conv1.register_forward_hook(hook_fn)
        model_t[1].layer1[0].conv1.register_forward_hook(hook_fn)
        model_t[1].layer2[0].conv1.register_forward_hook(hook_fn)
        model_t[1].layer3[0].conv1.register_forward_hook(hook_fn)
        model_t[1].layer4[0].conv1.register_forward_hook(hook_fn)
    else:
        model_t.conv1.register_forward_hook(hook_fn)
        model_t.layer1[0].conv1.register_forward_hook(hook_fn)
        model_t.layer2[0].conv1.register_forward_hook(hook_fn)
        model_t.layer3[0].conv1.register_forward_hook(hook_fn)
        model_t.layer4[0].conv1.register_forward_hook(hook_fn)



    # Ensure the model is in evaluation mode
    model_t.eval()

    # Perform a forward pass
    with torch.no_grad():
        input_batch = input_batch.cuda()
        _ = model_t(input_batch)

    # Function to plot feature maps
    def plot_feature_maps(feature_maps, layer, dir="./JPEG_DNN_analysis/images/", num_cols=8):
        fmap = feature_maps[layer].squeeze(0)  # Remove the batch dimension
        num_features = fmap.shape[0]
        num_rows = num_features // num_cols
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2))
        
        # for i, ax in enumerate(axes.flat):
        #     if i < num_features:
        #         # ax.imshow(fmap[i].cpu().numpy(), cmap='viridis')

        #         ax.imshow(fmap[i].cpu().numpy(),cmap='Greys',  interpolation='nearest')
        #         ax.axis('off')
        #     else:
        #         ax.remove()
        
        for i, ax in enumerate(fmap):
            plt.imshow(fmap[i].cpu().numpy(),cmap='gray')
            plt.savefig(dir+"_{}.png".format(i))

    # Plot feature maps for the layers
    # plot_feature_maps(feature_maps, model_t[1].conv1, dir= output_dir+"1")

    plot_feature_maps(feature_maps, model_t[1].layer1[0].conv1, dir= output_dir+"2")

    # plot_feature_maps(feature_maps, model_t[1].layer2[0].conv1, dir= output_dir+"3")

    # plot_feature_maps(feature_maps, model_t[1].layer3[0].conv1, dir= output_dir+"4")

    # plot_feature_maps(feature_maps, model_t[1].layer4[0].conv1, dir= output_dir+"5")



    exit(0)

    
    if opt.JPEG_enable:
        target_layers = [model_t[1].layer4]
    else:
        target_layers = model_t.layer4
    cam_run(model_t, image_path, output_dir, target_layers, opt)


    
    
    # model_t = model_t[0]
    # teacher_acc, _, _ = validate_image_analysis_without_frontend(val_loader, model_t, criterion_cls, opt)
    
    

    exit(0)

    steps = 5
    for eps in range(1, 10):
        # atk = torchattack.FGSM(model_t, eps=eps/255)
        alpha = (2.5*eps)/steps
        atk = torchattack.PGD(model_t, eps=eps/255, alpha=alpha/255, steps=steps, random_start=True)
        if opt.JPEG_enable:
            atk.set_normalization_used(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])
        else:
            atk.set_normalization_used(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        atk.set_model_training_mode(model_training=False)
        atk.set_mode_default()
        teacher_acc, _, _ = validate_attack(val_loader, model_t, criterion_cls, opt, atk)
        acc.append(teacher_acc.item())
        print('Attacked accuracy for {}/255: '.format(eps), teacher_acc.item())

        if opt.JPEG_enable:
            with torch.no_grad():
                quantizationTable = model_t[0].quantization_table.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                alpha = model_t[0].alpha.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
                print("Alpha --> Min: {:.2f}, Max: {:.2f}".format(alpha.min().item(), alpha.max().item()))
                print("Quantization Table --> Min: {:.2f}, Max: {:.2f}".format(quantizationTable.min().item(), quantizationTable.max().item()))
        
    acc = np.array(acc)
    dir = "./JPEG_DNN_analysis/Robustness/" + opt.path_t.split("/")[-2] + "_PGD.npy"
    np.save(dir, acc)
    # Load the array from the file
    # loaded_array = np.load(dir)


# # Function to plot feature maps
# def plot_feature_maps(feature_maps, layer, dir="../JPEG_DNN_analysis/images/", num_cols=8):
#     fmap = feature_maps[layer].squeeze(0)  # Remove the batch dimension
#     num_features = fmap.shape[0]
#     num_rows = num_features // num_cols
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2))
    
#     for i, ax in enumerate(axes.flat):
#         if i < num_features:
#             ax.imshow(fmap[i].cpu().numpy(), cmap='viridis')
#             ax.axis('off')
#         else:
#             ax.remove()
#     plt.savefig(dir+"_"+str(layer)+".png")

def cam_run(model, image_path, output_dir, target_layers, opt):

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    # target_layers = find_layer_types_recursive(model, [torch.nn.ReLU])
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    if opt.JPEG_enable:
        input_tensor = preprocess_image(rgb_img,
                                    mean=[0, 0, 0],
                                    std=[1/255., 1/255., 1/255.]).cuda()
    else:
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]).cuda()


    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputTarget(0)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = GradCAMPlusPlus
    method_name = "GradCAMPlusPlus"
    with cam_algorithm(model=model,
                       target_layers=target_layers) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, device="cuda")
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    os.makedirs(output_dir, exist_ok=True)

    cam_output_path = os.path.join(output_dir, f'{method_name}_cam.jpg')
    gb_output_path = os.path.join(output_dir, f'{method_name}_gb.jpg')
    cam_gb_output_path = os.path.join(output_dir, f'{method_name}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb)


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
