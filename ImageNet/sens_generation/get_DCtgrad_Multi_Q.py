import numpy as np
import torch
from torchvision import transforms
import torchvision
# from tqdm import tqdm
import torch.nn.functional as F
from utils import *
from torch import nn
from perc import Perc
import socket

hostname = socket.gethostname()

from model import get_model
import argparse

def sentivity_estimation(Y_sen_list, Cr_sen_list, Cb_sen_list):
    zigzag = get_zigzag()
    
    # Y_sen_list = np.load("./grad_Multi_Q/"+"Y_sen_list" + args.model + ".npy")
    # (49, 160000, 8, 8)
    lst_length = Y_sen_list.shape[1]
    Y_sen_img = np.zeros((Y_sen_list.shape[0], 64 ,lst_length))
    for q_idx in range(Y_sen_list.shape[0]):
        for i in range(8):
            for j in range(8):
                Y_sen_img[q_idx, zigzag[i,j]] = Y_sen_list[q_idx, :,i,j]
    del Y_sen_list
    Y_sens= np.sum((Y_sen_img)**2,-1) * (1/args.Nexample)
    np.save("SenMap_Multi_Q/Y"+args.model, Y_sens)
    del Y_sen_img
    
    # Cb_sen_list = np.load("./grad_Multi_Q/"+"Cr_sen_list" + args.model + ".npy")
    Cb_sen_img = np.zeros((Cb_sen_list.shape[0], 64 ,lst_length))
    for q_idx in range(Cb_sen_list.shape[0]):
        for i in range(8):
            for j in range(8):
                Cb_sen_img[q_idx, zigzag[i,j]] = Cb_sen_list[q_idx, :,i,j]
    del Cb_sen_list
    Cb_sens = np.sum((Cb_sen_img)**2,-1) * (1/args.Nexample)
    np.save("SenMap_Multi_Q/Cb"+args.model, Cb_sens)
    del Cb_sen_img

    # Cr_sen_list = np.load("./grad_Multi_Q/"+"Cb_sen_list" + args.model + ".npy")
    Cr_sen_img = np.zeros((Cr_sen_list.shape[0], 64,lst_length))
    for q_idx in range(Cr_sen_list.shape[0]):
        for i in range(8):
            for j in range(8):
                Cr_sen_img[q_idx, zigzag[i,j]] = Cr_sen_list[q_idx, :,i,j]
    Cr_sens= np.sum((Cr_sen_img)**2,-1) * (1/args.Nexample)
    np.save("SenMap_Multi_Q/Cr"+args.model, Cr_sens)
    del Cr_sen_img

    

class Senstivity_tools(object):
    def __init__(self, img_shape):
        super(Senstivity_tools, self).__init__()
        self.outter_block_size = 32
        self.inner_block_size = 8
        self.img_shape = img_shape
        self.pad_h, self.pad_w = self.pad_outter_block(self.img_shape[0]), self.pad_outter_block(self.img_shape[1])
        self.blocks_per_row = (self.pad_h + self.img_shape[0]) // self.outter_block_size
        self.blocks_per_col = (self.pad_w + self.img_shape[1]) // self.outter_block_size
        self.blocks_per_row_inner = self.outter_block_size // self.inner_block_size
        self.num_block_per_outter_block = self.blocks_per_row * self.blocks_per_col
        self.num_block_per_inner_block = (self.outter_block_size // self.inner_block_size) ** 2
    
    def pad_outter_block(self, height_width):
        return (self.outter_block_size - height_width % self.outter_block_size) % self.outter_block_size

    def blockify(self, input):
        image_padded = torch.nn.functional.pad(input, (0, self.pad_w, 0, self.pad_h))  # Pad the height and width

        # Step 2: Reshape the image to extract 32x32 blocks
        # We use unfold to split the height and width into 32x32 blocks
        blocks_32x32 = image_padded.unfold(2, self.outter_block_size, self.outter_block_size).unfold(3, self.outter_block_size, self.outter_block_size)  # (batch_size, channels, block_rows, block_cols, 32, 32)

        # Reshape into a 2D tensor with all 32x32 blocks
        blocks_32x32 = blocks_32x32.contiguous().view(-1, 3, self.num_block_per_outter_block, self.outter_block_size, self.outter_block_size)  # (256, 3, num_blocks, 32, 32)

        # Step 3: Further split each 32x32 block into 16 sub-blocks of 8x8
        blocks_8x8 = blocks_32x32.unfold(3, self.inner_block_size, self.inner_block_size).unfold(4, self.inner_block_size, self.inner_block_size)  # (256, 3, num_blocks, 4, 4, 8, 8)

        # Reshape into the desired output shape (256, 3, num_blocks, 16, 8, 8)
        blocks_8x8 = blocks_8x8.contiguous().view(-1, 3, self.num_block_per_outter_block, self.num_block_per_inner_block, self.inner_block_size, self.inner_block_size)
        
        return blocks_8x8

    def deblockify(self, input_DCT_block_batch):
        blocks_32x32_reconstructed = input_DCT_block_batch.view(-1, 3, self.num_block_per_outter_block, self.blocks_per_row_inner, self.blocks_per_row_inner, self.inner_block_size, self.inner_block_size).permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        blocks_32x32_reconstructed = blocks_32x32_reconstructed.view(-1, 3, self.num_block_per_outter_block, self.outter_block_size, self.outter_block_size)  # (batch_size, channels, num_blocks, 32, 32)
        image_reconstructed = blocks_32x32_reconstructed.view(-1, 3, self.blocks_per_row, self.blocks_per_col, self.outter_block_size, self.outter_block_size)
        image_reconstructed = image_reconstructed.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, 3, self.img_shape[0] + self.pad_h, self.img_shape[1] + self.pad_w)
        image_reconstructed = image_reconstructed[:, :, :self.img_shape[0], :self.img_shape[1]]
        
        return image_reconstructed
    
model_preprocess_parameters = {
            "ConvNeXt_base": [232, 224,transforms.InterpolationMode.BILINEAR],
            "ConvNeXt_large": [232, 224, transforms.InterpolationMode.BILINEAR],
            "ConvNeXt_tiny": [236, 224, transforms.InterpolationMode.BILINEAR],
            "ConvNeXt_small": [230, 224,transforms.InterpolationMode.BILINEAR],
            "mobilenet_v2_RV2": [232, 224, transforms.InterpolationMode.BILINEAR],
            "ViT_l_16" : [242, 224, transforms.InterpolationMode.BILINEAR],
            "ViT_l_32" : [242, 224, transforms.InterpolationMode.BILINEAR],
            "MaxVit_T" : [224, 224, transforms.InterpolationMode.BICUBIC],

            "Swin_T" : [232, 224, transforms.InterpolationMode.BICUBIC],
            "Swin_S" : [246, 224, transforms.InterpolationMode.BICUBIC],
            "Swin_B" : [238, 224, transforms.InterpolationMode.BICUBIC],

            "Swin_V2_T" : [260, 256, transforms.InterpolationMode.BICUBIC],
            "Swin_V2_S" : [260, 256, transforms.InterpolationMode.BICUBIC],
            "Swin_V2_B" : [272, 256, transforms.InterpolationMode.BICUBIC],

            "Regnet400mf_V2": [232, 224, transforms.InterpolationMode.BILINEAR],
            "Regnet800mf_V2": [232, 224, transforms.InterpolationMode.BILINEAR],
            "Regnet2gf_V2": [232, 224, transforms.InterpolationMode.BILINEAR],
            "Regnet8gf_V2": [232, 224, transforms.InterpolationMode.BILINEAR],

            
            "default" : [256, 224, transforms.InterpolationMode.BILINEAR],
        }
    

def main(args):
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    thr = args.Nexample
    model_name = args.model
    print("code run on", device)
    
    if model_name in list(model_preprocess_parameters.keys()):
        Resize_parameter = model_preprocess_parameters[model_name][0]
        centerCrop_parameter = model_preprocess_parameters[model_name][1]
        interpolation_method = model_preprocess_parameters[model_name][2]
    else:
        Resize_parameter = model_preprocess_parameters["default"][0]
        centerCrop_parameter = model_preprocess_parameters["default"][1]
        interpolation_method = model_preprocess_parameters["default"][2]  # Example: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'


    print("Resize : ", Resize_parameter , "Center Crop: ", centerCrop_parameter, "Interpolation Method: ", interpolation_method)

        
    Trans = [transforms.ToTensor(),
             transforms.Resize(Resize_parameter, interpolation=interpolation_method),
             transforms.CenterCrop(centerCrop_parameter),
             transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])
             ]


    if "emsolserv" in hostname:
        args.data_path = "/home/ahamsala/datasets/imagenet/"
    elif "multicomgpu.eng" in hostname:
        args.data_path = "/home/l44ye/datasets/"
    else:
        print("The current directory is ", args.data_path)

    transform = transforms.Compose(Trans)
    dataset = torchvision.datasets.ImageNet(root=args.data_path, split='train',
                                            transform=transform)
    resize256 = transforms.Resize(256)
    CenterCrop224 = transforms.CenterCrop(224)
    Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.Batch_size, shuffle=True, num_workers=8)

    pretrained_model = get_model(model_name)


    pretrained_model.to(device)
    pretrained_model.eval()
    Y_sen_list = []
    Cr_sen_list = []
    Cb_sen_list = []
    idx = 0
    criterion = nn.CrossEntropyLoss()
    # for data, target in tqdm(test_loader):
    sens_tool = Senstivity_tools((224,224,3))
    for data, target in Perc(test_loader):
        data, target = data.to(device), target.to(device)  # [0,225]
        img_shape = data.shape[-2:]

        input_DCT_block_batch = block_dct(sens_tool.blockify(rgb_to_ycbcr(data)))
        # input_DCT_block_batch = block_dct(blockify(rgb_to_ycbcr(data), 8))
        
        input_DCT_block_batch.requires_grad = True

        recoverd_img =  sens_tool.deblockify(block_idct(input_DCT_block_batch))
        # recoverd_img = deblockify(block_idct(input_DCT_block_batch), (img_shape[0], img_shape[1]))  # [-128, 127]
        
        norm_img = normalize(Scale2One(ycbcr_to_rgb(recoverd_img)))
        output = pretrained_model(norm_img)
        loss = criterion(output, target)
        pretrained_model.zero_grad()
        loss.backward()

        data_grad = input_DCT_block_batch.grad.permute(1, 2, 0, 3, 5, 4).detach().cpu().numpy()
        
        Y = data_grad[0].reshape(sens_tool.num_block_per_outter_block, -1, 8, 8)
        Y_sen_list.append(Y)
        Cb = data_grad[1].reshape(sens_tool.num_block_per_outter_block, -1, 8, 8)
        Cb_sen_list.append(Cb)
        Cr = data_grad[2].reshape(sens_tool.num_block_per_outter_block, -1, 8, 8)
        Cr_sen_list.append(Cr)
        idx += args.Batch_size
        if idx >= thr:
            break

    Y_sen_list = np.array(Y_sen_list).reshape(sens_tool.num_block_per_outter_block,-1,8,8) 
    print("Convert Y")
    Cr_sen_list = np.array(Cr_sen_list).reshape(sens_tool.num_block_per_outter_block,-1,8,8)
    print("Convert Cr")
    Cb_sen_list = np.array(Cb_sen_list).reshape(sens_tool.num_block_per_outter_block,-1,8,8)
    print("Convert Cb")
    print("")

    # np.save("./grad_Multi_Q/Y_sen_list" + model_name + ".npy",Y_sen_list)
    # np.save("./grad_Multi_Q/Cr_sen_list" + model_name + ".npy", Cr_sen_list)
    # np.save("./grad_Multi_Q/Cb_sen_list" + model_name + ".npy", Cb_sen_list)

    sentivity_estimation(Y_sen_list, Cr_sen_list, Cb_sen_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model, Batch_size, Nexample, resize')
    parser.add_argument('-model',type=str, default='alexnet', help='DNN model')
    parser.add_argument('-dev',type=str, default='cuda', help='device')
    parser.add_argument('-Batch_size', type=int, default=100,help='Number of examples in one batch')
    parser.add_argument('-Nexample',type=int, default=10000, help='Number of example')
    args = parser.parse_args()
    main(args)
    

