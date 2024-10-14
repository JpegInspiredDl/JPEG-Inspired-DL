import numpy as np
import torch
from torchvision import transforms
import torchvision
# from tqdm import tqdm
import torch.nn.functional as F
from utils import *
from torch import nn
from perc import Perc


from model import get_model
import argparse

def sentivity_estimation(args):
    zigzag = get_zigzag()
    
    Y_sen_list = np.load("./grad/"+"Y_sen_list" + args.model + ".npy")
    lst_length = Y_sen_list.shape[0]
    Y_sen_img = np.zeros((64,lst_length))
    
    for i in range(8):
        for j in range(8):
            Y_sen_img[zigzag[i,j]] = Y_sen_list[:,i,j]
    del Y_sen_list
    Y_sens= np.sum((Y_sen_img)**2,1) * (1/args.Nexample)
    np.save("SenMap/Y"+args.model, Y_sens)
    del Y_sen_img
    
    Cb_sen_list = np.load("./grad/"+"Cr_sen_list" + args.model + ".npy")
    Cb_sen_img = np.zeros((64,lst_length))
    for i in range(8):
        for j in range(8):
            Cb_sen_img[zigzag[i,j]] = Cb_sen_list[:,i,j]
    del Cb_sen_list
    Cb_sens = np.sum((Cb_sen_img)**2,1) * (1/args.Nexample)
    np.save("SenMap/Cb"+args.model, Cb_sens)
    del Cb_sen_img

    Cr_sen_list = np.load("./grad/"+"Cb_sen_list" + args.model + ".npy")
    Cr_sen_img = np.zeros((64,lst_length))
    for i in range(8):
        for j in range(8):
            Cr_sen_img[zigzag[i,j]] = Cr_sen_list[:,i,j]
    del Cr_sen_list
    Cr_sens= np.sum((Cr_sen_img)**2,1) * (1/args.Nexample)
    np.save("SenMap/Cr"+args.model, Cr_sens)
    del Cr_sen_img

    

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

    transform = transforms.Compose(Trans)
    dataset = torchvision.datasets.ImageNet(root="/home/l44ye/datasets/", split='train',
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
    for data, target in Perc(test_loader):
        data, target = data.to(device), target.to(device)  # [0,225]
        img_shape = data.shape[-2:]
        input_DCT_block_batch = block_dct(blockify(rgb_to_ycbcr(data), 8))
        input_DCT_block_batch.requires_grad = True
        recoverd_img = deblockify(block_idct(input_DCT_block_batch), (img_shape[0], img_shape[1]))  # [-128, 127]
        norm_img = normalize(Scale2One(ycbcr_to_rgb(recoverd_img)))
        output = pretrained_model(norm_img)
        loss = criterion(output, target)
        pretrained_model.zero_grad()
        loss.backward()
        data_grad = input_DCT_block_batch.grad.transpose(1, 0).detach().cpu().numpy()
        
        Y = data_grad[0].reshape(-1, 8, 8)
        Y_sen_list.append(Y)
        Cb = data_grad[1].reshape(-1, 8, 8)
        Cb_sen_list.append(Cb)
        Cr = data_grad[2].reshape(-1, 8, 8)
        Cr_sen_list.append(Cr)
        idx += args.Batch_size
        if idx >= thr:
            break

    Y_sen_list = np.array(Y_sen_list).reshape(-1,8,8) 
    print("Convert Y")
    Cr_sen_list = np.array(Cr_sen_list).reshape(-1,8,8)
    print("Convert Cr")
    Cb_sen_list = np.array(Cb_sen_list).reshape(-1,8,8)
    print("Convert Cb")
    print("")

    np.save("./grad/Y_sen_list" + model_name + ".npy",Y_sen_list)
    np.save("./grad/Cr_sen_list" + model_name + ".npy", Cr_sen_list)
    np.save("./grad/Cb_sen_list" + model_name + ".npy", Cb_sen_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model, Batch_size, Nexample, resize')
    parser.add_argument('-model',type=str, default='alexnet', help='DNN model')
    parser.add_argument('-dev',type=str, default='cuda', help='device')
    parser.add_argument('-Batch_size', type=int, default=100,help='Number of examples in one batch')
    parser.add_argument('-Nexample',type=int, default=10000, help='Number of example')
    args = parser.parse_args()
    main(args)
    sentivity_estimation(args)
