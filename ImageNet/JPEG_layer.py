import torch
import torch.nn.functional as F
import math
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from matplotlib.pyplot import figure
import torch.nn as nn
import os
from torchvision import transforms
from enum import Enum
# smallest_float32 = torch.finfo(torch.float32).tiny
smallest_float32 = 1e-5
from perc import Perc
largest_float32 = torch.finfo(torch.float32).max
# import matplotlib.pyplot as plt

class Channel(Enum):
    Y_Channel = 0
    Cb_channel = 1
    Cr_Channel = 2
    CbCr_Channel = 3

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

def second_max_fn(tensor):
    flattened_tensor = tensor.view(-1)
    max_value = torch.max(flattened_tensor)
    masked_tensor = torch.where(flattened_tensor == max_value, torch.tensor(float('-inf'), device=tensor.device), flattened_tensor)
    second_max_value = torch.max(masked_tensor)
    return second_max_value



def sentivity_estimation(Y_sen_list, Cb_sen_list, Cr_sen_list , num_images):
    zigzag = get_zigzag()
    lst_length = Y_sen_list.shape[0]
    Y_sen_img = np.zeros((64,lst_length))
    
    for i in range(8):
        for j in range(8):
            Y_sen_img[zigzag[i,j]] = Y_sen_list[:,i,j]
    Y_sens= np.sum((Y_sen_img)**2,1) * (1/num_images)
    
    Cb_sen_img = np.zeros((64,lst_length))
    for i in range(8):
        for j in range(8):
            Cb_sen_img[zigzag[i,j]] = Cb_sen_list[:,i,j]
    Cb_sens = np.sum((Cb_sen_img)**2,1) * (1/num_images)

    Cr_sen_img = np.zeros((64,lst_length))
    for i in range(8):
        for j in range(8):
            Cr_sen_img[zigzag[i,j]] = Cr_sen_list[:,i,j]
    Cr_sens= np.sum((Cr_sen_img)**2,1) * (1/num_images)

    return Y_sens, Cb_sens, Cr_sens

def get_sens(train_loader, pretrained_model, opt, mean, std, save=False):
    if torch.cuda.is_available():
        pretrained_model.to(device)
    pretrained_model.eval()
    Y_sen_list = []
    Cr_sen_list = []
    Cb_sen_list = []
    num_images = 0
    criterion = nn.CrossEntropyLoss()

    Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
    normalize = transforms.Normalize(mean=mean, std=std)

    block_dct_fn = block_dct_callable(opt.block_size)
    block_idct_fn = block_idct_callable(opt.block_size)

    for data, target in Perc(train_loader):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)  # [0,225]
        img_shape = data.shape[-2:]
        input_DCT_block_batch = block_dct_fn(blockify(rgb_to_ycbcr(data), 8))
        input_DCT_block_batch.requires_grad = True
        recoverd_img = deblockify(block_idct_fn(input_DCT_block_batch), (img_shape[0], img_shape[1]))  # [-128, 127]
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
        
        num_images += data.shape[0]
        if num_images >= opt.Nexample:
            break
    Y_sen_list = np.array(Y_sen_list).reshape(-1,8,8) 
    print("")
    print("Convert Y")
    Cr_sen_list = np.array(Cr_sen_list).reshape(-1,8,8)
    print("Convert Cr")
    Cb_sen_list = np.array(Cb_sen_list).reshape(-1,8,8)
    print("Convert Cb")
    print("")


    Y_sens, Cb_sens, Cr_sens = sentivity_estimation(Y_sen_list, Cb_sen_list, Cr_sen_list , num_images)
    if not os.path.exists(opt.sensitvity_dir):
        # Create a new directory because it does not exist
        os.makedirs(opt.sensitvity_dir)
    print("The new directory is created!")
    if save:
        # np.save("./Senstivity_calc/grad/Y_sen_list_" + opt.model + ".npy",Y_sen_list)
        # np.save("./Senstivity_calc/grad/Cr_sen_list_" + opt.model + ".npy", Cr_sen_list)
        # np.save("./Senstivity_calc/grad/Cb_sen_list_" + opt.model + ".npy", Cb_sen_list)
        np.save(opt.sensitvity_dir + "Y_"+opt.model, Y_sens)
        np.save(opt.sensitvity_dir + "Cb_"+opt.model, Cb_sens)
        np.save(opt.sensitvity_dir + "Cr_"+opt.model, Cr_sens)

    return Y_sens, Cb_sens, Cr_sens



def normalize_2d(arr, factor):
    if not torch.is_tensor(factor) and factor == 0 :
        factor, _ = torch.max(torch.max(arr, dim=2).values, dim=1)
        factor = factor.unsqueeze(1).unsqueeze(2)
    arr = arr/factor
    return arr, factor

def normalize(arr, factor):
    if factor == 0:
        factor = torch.max(arr)
    arr = arr/factor
    return arr, factor

# Define your neural network
class JPEG_layer(nn.Module):

    def construct(self, opt):
        self.num_bit = opt.num_bit
        self.Q_inital = opt.Q_inital
        self.batch_size = opt.batch_size
        self.block_size = opt.block_size
        self.inner_block_size = opt.block_size
        # self.num_channels = self.img_shape[-1]
        self.num_channels = 3
        self.num_block = int((self.img_shape[0]*self.img_shape[1])/(self.block_size**2))
        self.min_Q_Step = opt.min_Q_Step
        self.max_Q_Step = opt.max_Q_Step
        self.num_non_zero_q = opt.num_non_zero_q
        self.JEPG_alpha = opt.JEPG_alpha
        self.q_max =opt.q_max
        self.q_min =opt.q_min
        self.hardness_matching=opt.hardness_matching
        self.hardness = opt.hardness
        self.outter_block_size = opt.outter_block_size

        self.alpha_scaling = opt.alpha_scaling
        self.alpha_fixed = opt.alpha_fixed
        self.zero_dc = opt.zero_dc

        

        self.pad_h, self.pad_w = self.pad_outter_block(self.img_shape[0]), self.pad_outter_block(self.img_shape[1])
        self.blocks_per_row = (self.pad_h + self.img_shape[0]) // self.outter_block_size
        self.blocks_per_col = (self.pad_w + self.img_shape[1]) // self.outter_block_size
        self.blocks_per_row_inner = self.outter_block_size // self.inner_block_size
        self.num_block_per_outter_block = self.blocks_per_row * self.blocks_per_col
        self.num_block_per_inner_block = (self.outter_block_size // self.inner_block_size) ** 2

        if os.path.isfile(opt.sensitvity_dir + "Y" + opt.model + ".npy") and opt.initial_Q_w_sensitivity:
            print(f"Loading the correct sensitivity from ", f"{opt.sensitvity_dir}Y{opt.model}.npy")
            
            Y_sens  = inverse_zigzag_2d(torch.from_numpy(np.load(opt.sensitvity_dir + "Y" + opt.model + ".npy")))
            Cb_sens = inverse_zigzag_2d(torch.from_numpy(np.load(opt.sensitvity_dir + "Cb"+ opt.model + ".npy")))
            Cr_sens = inverse_zigzag_2d(torch.from_numpy(np.load(opt.sensitvity_dir + "Cr"+ opt.model + ".npy")))
            # torch.Size([batch_size, num_channel, num_block_per_outter_block , num_block_per_inner_block , block_size, block_size, 1])
            Y_sens               =  1 / Y_sens 
            CbCr_sens            =  2 / (Cb_sens + Cr_sens)
            _    , factor        = normalize_2d(Y_sens   , 0)
            factor               = factor / self.q_max
            self.Y_sens    , _   = normalize_2d(Y_sens   , factor)
            self.CbCr_sens , _   = normalize_2d(CbCr_sens, factor)
            print("JPEG layer ==> load sensitivity and Normalize it")
            
        
        elif os.path.isfile(opt.dir_initial_abs_avg) and opt.initial_Q_w_abs_avg:
            tables_abs_sum = torch.load(opt.dir_initial_abs_avg)
            abs_avg_Y  = tables_abs_sum["abs_avg_Y"].cpu()
            abs_avg_CbCr = tables_abs_sum["abs_avg_CbCr"].cpu()
            self.Y_sens = 2 * abs_avg_Y / (2**(self.num_bit - 1)) ** (0.5)
            self.CbCr_sens = 2 * abs_avg_CbCr / (2**(self.num_bit - 1)) ** (0.5)
            if self.q_min == 0:
                self.Y_sens[0, 0] = zigzag(self.Y_sens)[1]
                self.CbCr_sens[0, 0] = zigzag(self.CbCr_sens)[1]
            print("JPEG layer ==> load absloute average and Normalize it by")
        
        else:
            print("JPEG layer ==> No senstivity generated ...")
            print("JPEG layer ==> Initialize the Q tables with ones ...")       

        self.std_width_Y = opt.std_width_Y
        self.std_width_CbCr = opt.std_width_CbCr


        self.num_non_zero_q_on_1_side = math.floor(self.num_non_zero_q/2)
        self.q_idx = torch.arange(0, self.num_non_zero_q)
        

        # ---------------------------------------------------------------------------------------
        '''
            In this part, we clamp all the cofficients to the same level, which is based on number of bits.
        '''

        self.num_bits_Y = torch.ones((self.block_size, self.block_size), dtype=torch.float32) * (opt.num_bit - 1)
        self.num_bits_CbCr =  torch.ones((self.block_size, self.block_size), dtype=torch.float32) * (opt.num_bit - 1)
        
        self.low_level_Y  =  (-2**(self.num_bits_Y) + 1).view(1, 1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 
        self.high_level_Y =  ( 2**(self.num_bits_Y)).view(1, 1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 

        self.low_level_CbCr  =  (-2**(self.num_bits_CbCr) + 1).view(1, 1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 
        self.high_level_CbCr =  ( 2**(self.num_bits_CbCr)).view(1, 1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 


        # ---------------------------------------------------------------------------------------
        '''
        In this part, we clamp all the cofficients to the same level, which is based on the standard deivation.
        '''

        # self.mask_Y = self.std_width_Y * self.Y_level * torch.ones((self.block_size, self.block_size), dtype=torch.float32, device=self.Y_level.device) 
        # self.mask_CbCr = self.std_width_CbCr * self.CbCr_level * torch.ones((self.block_size, self.block_size), dtype=torch.float32, device=self.Y_level.device) 
        
        # self.low_level_Y  =  -1 * self.mask_Y.view(1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 
        # self.high_level_Y =  self.mask_Y.view(1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 

        # self.low_level_CbCr  =  -1 * self.mask_CbCr.view(1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 
        # self.high_level_CbCr =  self.mask_CbCr.view(1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 

        
        # ---------------------------------------------------------------------------------------
        

        # ---------------------------------------------------------------------------------------
        '''
        In this part, we clamp only the dc with different std_width_Y and std_width_CbCr factor and we allocate for
        the AC cofficients 11 bits (no clampping).
        '''

        # self.num_bits_Y = torch.ones((self.block_size, self.block_size), dtype=torch.float32) * (opt.num_bit - 1)
        # self.num_bits_CbCr =  torch.ones((self.block_size, self.block_size), dtype=torch.float32) * (opt.num_bit - 1)
        # self.low_level_Y  =  (-2**(self.num_bits_Y) + 1).view(1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 
        # self.high_level_Y =  ( 2**(self.num_bits_Y)).view(1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 

        # self.low_level_CbCr  =  (-2**(self.num_bits_CbCr) + 1).view(1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 
        # self.high_level_CbCr =  ( 2**(self.num_bits_CbCr)).view(1, 1, 1, self.block_size, self.block_size, 1) # Batch, Channel, # of Blocks, B, B, 1 


        # self.low_level_Y[0,0,0,0,0,0] = -1 * self.std_width_Y * self.Y_level
        # self.high_level_Y[0,0,0,0,0,0] = self.std_width_Y * self.Y_level

        # self.low_level_CbCr[0,0,0,0,0,0]    = -1 * self.std_width_CbCr * self.CbCr_level
        # self.high_level_CbCr[0,0,0,0,0,0]   = self.std_width_CbCr * self.CbCr_level
        
        # ---------------------------------------------------------------------------------------
        


    
    def __init__(self, opt, img_shape, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], analysis=False):
        super(JPEG_layer, self).__init__()
        self.JPEG_alpha_trainable = opt.JPEG_alpha_trainable
        self.img_shape = img_shape
        self.analysis = analysis
        self.jpeg_layers = opt.jpeg_layers

        self.construct(opt)

        # torch.Size([batch_size, num_channel, num_block_per_outter_block , num_block_per_inner_block , block_size, block_size, 1])
        self.hardness_Y  = opt.hardness * torch.ones((1, 1, self.num_block_per_outter_block, 1 , opt.block_size, opt.block_size, 1), dtype=torch.float32)
        self.hardness_CbCr  = opt.hardness * torch.ones((1, 1, self.num_block_per_outter_block, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32)


        if torch.cuda.is_available():
            self.q_idx = self.q_idx.cuda()
            self.high_level_Y =  self.high_level_Y.cuda()
            self.low_level_Y  = self.low_level_Y.cuda()

            self.high_level_CbCr = self.high_level_CbCr.cuda()
            self.low_level_CbCr  = self.low_level_CbCr.cuda()

            self.hardness_Y  = self.hardness_Y.cuda()
            self.hardness_CbCr = self.hardness_CbCr.cuda()

        if self.jpeg_layers > 1:
            self.lum_qtable = torch.ones((opt.jpeg_layers, 1, 1, self.num_block_per_outter_block, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32)
            self.chrom_qtable = torch.ones((opt.jpeg_layers, 1, 1, self.num_block_per_outter_block, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32)
            self.alpha_lum = torch.ones((opt.jpeg_layers, 1, 1, self.num_block_per_outter_block, 1, 1, 1, 1), dtype=torch.float32)
            self.alpha_chrom = torch.ones((opt.jpeg_layers, 1, 1, self.num_block_per_outter_block, 1, 1, 1, 1), dtype=torch.float32)
        else:
            self.lum_qtable = torch.ones((1, 1, self.num_block_per_outter_block, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32)
            self.chrom_qtable = torch.ones((1, 1, self.num_block_per_outter_block, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32)
            self.alpha_lum = torch.ones((1, 1, self.num_block_per_outter_block, 1, 1, 1, 1), dtype=torch.float32)
            self.alpha_chrom = torch.ones((1, 1, self.num_block_per_outter_block, 1, 1, 1, 1), dtype=torch.float32)


        nn.init.constant_(self.lum_qtable , self.Q_inital)
        nn.init.constant_(self.chrom_qtable , self.Q_inital)
        nn.init.constant_(self.alpha_lum, self.JEPG_alpha)
        nn.init.constant_(self.alpha_chrom, self.JEPG_alpha)

        if  opt.initial_Q_w_sensitivity:
            print("JPEG layer ==> ** Q table ** is initialized by Sensitivity using Inverse Normalization method ... ")
            self.lum_qtable = self.lum_qtable * self.Y_sens.view(1,1, self.num_block_per_outter_block,1,self.block_size, self.block_size, 1)
            self.chrom_qtable = self.chrom_qtable * self.CbCr_sens.view(1,1, self.num_block_per_outter_block,1,self.block_size, self.block_size, 1)
        elif  opt.initial_Q_w_abs_avg:
            print("JPEG layer ==> ** Q table ** is initialized by absloute average using  2|x| / sqrt(2^(n-1)) ... ")
            self.lum_qtable = self.lum_qtable * self.Y_sens.view(1,1,1,1,self.block_size, self.block_size, 1)
            self.chrom_qtable = self.chrom_qtable * self.CbCr_sens.view(1,1, 1,1,self.block_size, self.block_size, 1)
        else:
            print("JPEG layer ==> ** Q table ** is initialized by ONES ... ")


        if self.alpha_scaling:
            print("JPEG layer ==> ** Alpha ** is scaled by Sensitvity ... ")
            self.alpha_lum = self.alpha_lum / self.Y_sens.view(1,1,1,self.block_size, self.block_size, 1)
            self.alpha_chrom = self.alpha_chrom / self.CbCr_sens.view(1,1,1,self.block_size, self.block_size, 1)
            self.alpha_lum.copy_(torch.clamp(self.alpha_lum, min=smallest_float32, max=largest_float32))
            self.alpha_chrom.copy_(torch.clamp(self.alpha_chrom, min=smallest_float32, max=largest_float32))
        elif self.alpha_fixed:
            print("JPEG layer ==> ** Alpha ** is fixd and not trainable ... ")


        # QT_Y = torch.tensor(quantizationTable(QF=50, Luminance=True))
        # QT_C = torch.tensor(quantizationTable(QF=50, Luminance=False))
        # self.initial_values = torch.stack([QT_Y, QT_C, QT_C], dim=0)

        self.block_idct = block_idct_callable(opt.block_size)
        self.block_dct = block_dct_callable(opt.block_size)

        
        # self.rgb_to_ycbcr = rgb_to_ycbcr_batch()
        # self.ycbcr_to_rgb = ycbcr_to_rgb_batch()


        self.lum_qtable = nn.Parameter(self.lum_qtable)
        self.chrom_qtable = nn.Parameter(self.chrom_qtable)
        
        if opt.JPEG_alpha_trainable:
            print("JPEG layer ==> ** Alpha ** is trainable")
            self.alpha_lum = nn.Parameter(self.alpha_lum)
            self.alpha_chrom = nn.Parameter(self.alpha_chrom)
        else:
            print("JPEG layer ==> ** Alpha ** is not trainable")
            if torch.cuda.is_available():
                self.alpha_lum = self.alpha_lum.cuda()
                self.alpha_chrom = self.alpha_chrom.cuda()

        self.Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.register_forward_pre_hook(self.reinitialize_q_table_alpha)

        # self.lum_qtable.register_hook(self.gradient_hook)
        # self.chrom_qtable.register_hook(self.gradient_hook)

    # def gradient_hook(self, grad):
        # grad.view(8,8)
        # grad[0,0,0,0,0,0] = 0.0
        # assert grad[0,0,0,0,0,0] != 0.0 , f"Indexes out of bounds: invalid values found {grad[0,0,0,0,0,0]}"
        # print(grad[0,0,0,0,0,0])
        # return grad

    def forward_zero_DC(self, input_RGB):
        # mean_per_image = input.mean(dim=(2, 3), keepdim=True)  # Shape (N, C, 1, 1)
        # input_RGB = denormalize(input_RGB, self.mean, self.std) * 255

        input = input_RGB - 128
        input_DCT_block_batch = self.block_dct(blockify(rgb_to_ycbcr(input), self.block_size))

        input_lum   = input_DCT_block_batch[:, 0:1, ...]
        input_chrom = input_DCT_block_batch[:, 1:3, ...]

        # std_lum = input_lum.std()
        # std_chrom = input_chrom.std()
        # input_lum[:,:,:,0,0] = torch.clamp(input_lum[:,:,:,0,0], min=-2*std_lum, max=2*std_lum)
        # input_chrom[:,:,:,0,0] = torch.clamp(input_chrom[:,:,:,0,0], min=-2*std_chrom, max=2*std_chrom)

        input_lum[:,:,:,0,0] = torch.clamp(input_lum[:,:,:,0,0], min=0, max=0)
        input_chrom[:,:,:,0,0] = torch.clamp(input_chrom[:,:,:,0,0], min=0, max=0)

        estimated_reconstructed_space = torch.cat((input_lum, input_chrom), 1)
        
        norm_img =  ycbcr_to_rgb(deblockify(self.block_idct(estimated_reconstructed_space), (self.img_shape[0], self.img_shape[1])))
        norm_img += 128
        # norm_img =  (norm_img - norm_img.min())
        # norm_img =  (norm_img/norm_img.max()) * 255.0
        # Here I am doing the same effect of a tensor by using Scale2One then normalize using the standard normalization
        norm_img = self.normalize(self.Scale2One(norm_img))    
        return norm_img

    def forward(self, x, analysis=False):
        if self.zero_dc:
            return self.forward_zero_DC(x)
        elif self.jpeg_layers > 1:
            return self.forward_jpeg_layers(x, analysis)
        else:
            return self.forward_jpeg_layer(x, analysis)

    def Qd_block(self, input_lum, input_chrom, q_table_idx):
        idx_lum   = torch.round(input_lum / self.lum_qtable[q_table_idx])
        idx_chrom = torch.round(input_chrom / self.chrom_qtable[q_table_idx].expand(1, 2, 1, self.block_size, self.block_size, 1))
        
        idx_lum   =  torch.clamp(idx_lum   - self.num_non_zero_q_on_1_side, min=self.low_level_Y, max=self.high_level_Y - self.num_non_zero_q)
        idx_chrom =  torch.clamp(idx_chrom - self.num_non_zero_q_on_1_side, min=self.low_level_CbCr, max=self.high_level_CbCr - self.num_non_zero_q)

        idx_lum = idx_lum.expand(-1, -1, -1, -1, -1, self.num_non_zero_q) + self.q_idx
        idx_chrom = idx_chrom.expand(-1, 2, -1, -1, -1, self.num_non_zero_q) + self.q_idx
        
        iq_lum = idx_lum * self.lum_qtable[q_table_idx]
        iq_chrom = idx_chrom * self.chrom_qtable[q_table_idx]

        distortion_MSE_mask_lum = F.mse_loss(iq_lum, input_lum.expand(-1, -1, -1 ,-1 ,-1 , self.num_non_zero_q), reduction='none')
        distortion_MSE_mask_chrom = F.mse_loss(iq_chrom, input_chrom.expand(-1, -1, -1 ,-1 ,-1 , self.num_non_zero_q), reduction='none')

        cmpf_mask_lum = F.softmax(-self.alpha_lum[q_table_idx] * distortion_MSE_mask_lum, dim=-1) # shape: [bs,ch,B,size,size,5]
        cmpf_mask_chrom = F.softmax(-self.alpha_chrom[q_table_idx].expand(-1, 2, -1, -1, -1, 1) * distortion_MSE_mask_chrom , dim=-1) # shape: [bs,ch,B,size,size,5] 

        estimated_reconstructed_space_lum = torch.sum(cmpf_mask_lum * iq_lum , -1)
        estimated_reconstructed_space_chrom  = torch.sum(cmpf_mask_chrom * iq_chrom, -1)

        return estimated_reconstructed_space_lum.unsqueeze(-1), estimated_reconstructed_space_chrom.unsqueeze(-1)

    def forward_jpeg_layers(self, input_RGB, analysis=False):
        # Check for NaN values and assert
        # assert not torch.isnan(self.lum_qtable).any(), "lum_qtable contains NaN values"
        # assert not torch.isnan(self.chrom_qtable).any(), "chrom_qtable contains NaN values"
        # assert not torch.isnan(self.alpha_lum).any(), "alpha_lum contains NaN values"
        # assert not torch.isnan(self.alpha_chrom).any(), "alpha_chrom contains NaN values"
    
        
        # mean_per_image = input.mean(dim=(2, 3), keepdim=True)  # Shape (N, C, 1, 1)
        input = input_RGB - 128
        input_DCT_block_batch = self.block_dct(blockify(rgb_to_ycbcr(input), self.block_size)).unsqueeze(-1)

        input_lum   = input_DCT_block_batch[:, 0:1, ...]
        input_chrom = input_DCT_block_batch[:, 1:3, ...]

        for q_table_idx in range(self.jpeg_layers):
            input_lum, input_chrom = self.Qd_block(input_lum, input_chrom, q_table_idx)

        estimated_reconstructed_space_lum, estimated_reconstructed_space_chrom = input_lum.squeeze(-1), input_chrom.squeeze(-1)
        estimated_reconstructed_space = torch.cat((estimated_reconstructed_space_lum, estimated_reconstructed_space_chrom), 1)
        norm_img =  ycbcr_to_rgb(deblockify(self.block_idct(estimated_reconstructed_space), (self.img_shape[0], self.img_shape[1])))
        norm_img += 128

        # Here I am doing the same effect of a tensor by using Scale2One then normalize using the standard normalization
        # assert torch.all(norm_img >= -2 ), \
        #     f"Indexes out of bounds: invalid values found {norm_img[~(norm_img >= -2)]} {norm_img.min()}"

        # if norm_img.min() < 0:
        #     print(norm_img.min().item(), norm_img.max().item())
        
        # norm_img = torch.clamp(norm_img, min=0.0, max=255.0)
        norm_img = self.normalize(self.Scale2One(norm_img))   
        if analysis:
            return norm_img, input_DCT_block_batch, estimated_reconstructed_space
        else:
            return norm_img
      
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
    
    def forward_jpeg_layer(self, input_RGB, analysis=False):
        # Check for NaN values and assert
        # assert not torch.isnan(self.lum_qtable).any(), "lum_qtable contains NaN values"
        # assert not torch.isnan(self.chrom_qtable).any(), "chrom_qtable contains NaN values"
        # assert not torch.isnan(self.alpha_lum).any(), "alpha_lum contains NaN values"
        # assert not torch.isnan(self.alpha_chrom).any(), "alpha_chrom contains NaN values"
    
        # mean_per_image = input.mean(dim=(2, 3), keepdim=True)  # Shape (N, C, 1, 1)
        # batch_size, h, w = input_RGB.shape[0], input_RGB.shape[2], input_RGB.shape[3]
        input = input_RGB - 128

        input_DCT_block_batch = self.block_dct(self.blockify(rgb_to_ycbcr(input))).unsqueeze(-1)
        # Speical case of the implemented general case
        # input_DCT_block_batch = self.block_dct(blockify(rgb_to_ycbcr(input), self.block_size)).unsqueeze(-1)

        input_lum   = input_DCT_block_batch[:, 0:1, ...]
        input_chrom = input_DCT_block_batch[:, 1:3, ...]

        idx_lum   = torch.round(input_lum / self.lum_qtable)
        idx_chrom = torch.round(input_chrom / self.chrom_qtable.expand(1, 2, self.num_block_per_outter_block, 1, self.block_size, self.block_size, 1))
        
        # idx_lum   =  torch.clamp(idx_lum - self.num_non_zero_q_on_1_side, min=self.low_level, max=self.high_level - self.num_non_zero_q)
        # idx_chrom  =  torch.clamp(idx_chrom - self.num_non_zero_q_on_1_side, min=self.low_level, max=self.high_level - self.num_non_zero_q)

        # idx_lum   =  torch.clamp(idx_lum - self.num_non_zero_q_on_1_side, min=-self.std_width_Y*self.Y_level, max=self.std_width_Y*self.Y_level - self.num_non_zero_q)
        # idx_chrom  =  torch.clamp(idx_chrom - self.num_non_zero_q_on_1_side, min=-self.std_width_CbCr*self.CbCr_level, max=self.std_width_CbCr - self.num_non_zero_q)

        idx_lum   =  torch.clamp(idx_lum   - self.num_non_zero_q_on_1_side, min=self.low_level_Y, max=self.high_level_Y - self.num_non_zero_q)
        idx_chrom =  torch.clamp(idx_chrom - self.num_non_zero_q_on_1_side, min=self.low_level_CbCr, max=self.high_level_CbCr - self.num_non_zero_q)

        idx_lum = idx_lum.expand(-1, -1, -1, -1, -1, -1, self.num_non_zero_q) + self.q_idx
        idx_chrom = idx_chrom.expand(-1, 2, -1, -1, -1, -1, self.num_non_zero_q) + self.q_idx
        
        iq_lum = idx_lum.detach() * self.lum_qtable[0]
        iq_chrom = idx_chrom.detach() * self.chrom_qtable[0]

        distortion_MSE_mask_lum = F.mse_loss(iq_lum, input_lum.expand(-1, -1, -1, -1 ,-1 ,-1 , self.num_non_zero_q), reduction='none')
        distortion_MSE_mask_chrom = F.mse_loss(iq_chrom, input_chrom.expand(-1, -1, -1, -1 ,-1 ,-1 , self.num_non_zero_q), reduction='none')
        
        cmpf_mask_lum = F.softmax(-self.alpha_lum * distortion_MSE_mask_lum, dim=-1) # shape: [bs,ch,B,size,size,5]
        cmpf_mask_chrom = F.softmax(-self.alpha_chrom.expand(-1, 2, -1, -1, -1, -1, -1) * distortion_MSE_mask_chrom , dim=-1) # shape: [bs,ch,B,size,size,5]
        
        # Q_d
        estimated_reconstructed_space_lum = torch.sum(cmpf_mask_lum * iq_lum , -1)
        estimated_reconstructed_space_chrom  = torch.sum(cmpf_mask_chrom * iq_chrom, -1)
        estimated_reconstructed_space = torch.cat((estimated_reconstructed_space_lum, estimated_reconstructed_space_chrom), 1)

        norm_img =  ycbcr_to_rgb(self.deblockify(self.block_idct(estimated_reconstructed_space)))
        # Speical case of the implemented general case
        # norm_img =  ycbcr_to_rgb(deblockify(self.block_idct(estimated_reconstructed_space), (self.img_shape[0], self.img_shape[1])))

        norm_img += 128

        # Here I am doing the same effect of a tensor by using Scale2One then normalize using the standard normalization
        # assert torch.all(norm_img >= -2 ), \
        #     f"Indexes out of bounds: invalid values found {norm_img[~(norm_img >= -2)]} {norm_img.min()}"

        # if norm_img.min() < 0:
        #     print(norm_img.min().item(), norm_img.max().item())
        
        # norm_img = torch.clamp(norm_img, min=0.0, max=255.0)
        norm_img = self.normalize(self.Scale2One(norm_img))   
        if analysis:
            return norm_img, input_DCT_block_batch, estimated_reconstructed_space
        else:
            return norm_img
        
    def reinitialize_q_table_alpha(self, model, input):
        with torch.no_grad():
            if self.min_Q_Step == 0:
                self.lum_qtable.copy_(torch.clamp(self.lum_qtable, min=smallest_float32, max=self.max_Q_Step))
                self.chrom_qtable.copy_(torch.clamp(self.chrom_qtable, min=smallest_float32, max=self.max_Q_Step))
            else:
                self.lum_qtable.copy_(torch.clamp(self.lum_qtable, min=self.min_Q_Step, max=self.max_Q_Step))
                self.chrom_qtable.copy_(torch.clamp(self.chrom_qtable, min=self.min_Q_Step, max=self.max_Q_Step))
            
            if self.hardness_matching:
                self.alpha_lum.copy_(self.hardness / (self.lum_qtable ** 2))
                self.alpha_chrom.copy_(self.hardness / (self.chrom_qtable ** 2))
                self.alpha_lum.copy_(torch.clamp(self.alpha_lum, min=smallest_float32, max=largest_float32))
                self.alpha_chrom.copy_(torch.clamp(self.alpha_chrom, min=smallest_float32, max=largest_float32))

            if self.JPEG_alpha_trainable:
                self.alpha_lum.copy_(torch.clamp(self.alpha_lum, min=smallest_float32, max=largest_float32))
                self.alpha_chrom.copy_(torch.clamp(self.alpha_chrom, min=smallest_float32, max=largest_float32))

def zigzag(matrix):
    zigzag = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                           [2, 4, 7, 13, 16, 26, 29, 42],
                           [3, 8, 12, 17, 25, 30, 41, 43],
                           [9, 11, 18, 24, 31, 40, 44, 53],
                           [10, 19, 23, 32, 39, 45, 52, 54],
                           [20, 22, 33, 38, 46, 51, 55, 60],
                           [21, 34, 37, 47, 50, 56, 59, 61],
                           [35, 36, 48, 49, 57, 58, 62, 63]])
    
    

    # Get the shape of the matrix
    matrix_size = np.shape(matrix)

    # Calculate the size of the vector
    vector_size = np.shape(zigzag)

    # Check if the matrix size matches the vector size
    if matrix_size != vector_size:
        raise ValueError("The matrix size does not match the vector size.")

    # Create an empty vector to store the values
    vector = np.zeros(matrix_size[0] * matrix_size[1])

    # Iterate over each element in the matrix and place it in the corresponding position in the vector
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            index = zigzag[i, j]
            vector[index] = matrix[i, j]

    return vector

def inverse_zigzag_2d(vector):
    zigzag = torch.tensor([[0, 1, 5, 6, 14, 15, 27, 28], \
                           [2, 4, 7, 13, 16, 26, 29, 42], \
                           [3, 8, 12, 17, 25, 30, 41, 43], \
                           [9, 11, 18, 24, 31, 40, 44, 53], \
                           [10, 19, 23, 32, 39, 45, 52, 54], \
                           [20, 22, 33, 38, 46, 51, 55, 60], \
                           [21, 34, 37, 47, 50, 56, 59, 61], \
                           [35, 36, 48, 49, 57, 58, 62, 63]])

    # Get the shape of the vector
    vector_shape = vector.size()

    # Calculate the size of the 2D matrix
    matrix_size = zigzag.size()

    # Create an empty matrix to store the values
    matrix = torch.zeros(vector.shape[0], 8, 8)

    # Iterate over each element in the vector and place it in the corresponding position in the matrix
    for q_idx in range(vector.shape[0]):
        for i in range(matrix_size[0]):
            for j in range(matrix_size[1]):
                index = zigzag[i, j]
                matrix[q_idx, i, j] = vector[q_idx, index]

    return matrix

def inverse_zigzag(vector):
    zigzag = torch.tensor([[0, 1, 5, 6, 14, 15, 27, 28], \
                           [2, 4, 7, 13, 16, 26, 29, 42], \
                           [3, 8, 12, 17, 25, 30, 41, 43], \
                           [9, 11, 18, 24, 31, 40, 44, 53], \
                           [10, 19, 23, 32, 39, 45, 52, 54], \
                           [20, 22, 33, 38, 46, 51, 55, 60], \
                           [21, 34, 37, 47, 50, 56, 59, 61], \
                           [35, 36, 48, 49, 57, 58, 62, 63]])

    # Get the shape of the vector
    vector_shape = vector.size()

    # Calculate the size of the 2D matrix
    matrix_size = zigzag.size()

    # Create an empty matrix to store the values
    matrix = torch.zeros(matrix_size)

    # Iterate over each element in the vector and place it in the corresponding position in the matrix
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            index = zigzag[i, j]
            matrix[i, j] = vector[index]

    return matrix


def get_average_model_magnitude(model):
    total_magnitude = 0.0
    num_parameters = 0
    for param in model.parameters():
        param_magnitude = torch.mean(torch.abs(param)).item()
        total_magnitude += param_magnitude
        num_parameters += 1
    if num_parameters == 0:
        return 0.0
    return total_magnitude / num_parameters


def get_max_model_magnitude(model):
    max_magnitude = 0.0
    for param in model.parameters():
        param_max = torch.max(torch.abs(param)).item()
        if param_max > max_magnitude:
            max_magnitude = param_max
    return max_magnitude


def quantizationTable(QF=50, Luminance=True):
    #  Luminance quantization table
    #  Standard
    # * 16 11 10 16 24  40  51  61
    # * 12 12 14 19 26  58  60  55
    # * 14 13 16 24 40  57  69  56
    # * 14 17 22 29 51  87  80  62
    # * 18 22 37 56 68  109 103 77
    # * 24 35 55 64 81  104 113 92
    # * 49 64 78 87 103 121 120 101
    # * 72 92 95 98 112 100 103 99

    quantizationTableData = np.ones((8, 8), dtype=np.float32)

    if QF == 100:
        # print(quantizationTableData)
        return quantizationTableData

    if Luminance == True:  # Y channel
        quantizationTableData[0][0] = 16
        quantizationTableData[0][1] = 11
        quantizationTableData[0][2] = 10
        quantizationTableData[0][3] = 16
        quantizationTableData[0][4] = 24
        quantizationTableData[0][5] = 40
        quantizationTableData[0][6] = 51
        quantizationTableData[0][7] = 61
        quantizationTableData[1][0] = 12
        quantizationTableData[1][1] = 12
        quantizationTableData[1][2] = 14
        quantizationTableData[1][3] = 19
        quantizationTableData[1][4] = 26
        quantizationTableData[1][5] = 58
        quantizationTableData[1][6] = 60
        quantizationTableData[1][7] = 55
        quantizationTableData[2][0] = 14
        quantizationTableData[2][1] = 13
        quantizationTableData[2][2] = 16
        quantizationTableData[2][3] = 24
        quantizationTableData[2][4] = 40
        quantizationTableData[2][5] = 57
        quantizationTableData[2][6] = 69
        quantizationTableData[2][7] = 56
        quantizationTableData[3][0] = 14
        quantizationTableData[3][1] = 17
        quantizationTableData[3][2] = 22
        quantizationTableData[3][3] = 29
        quantizationTableData[3][4] = 51
        quantizationTableData[3][5] = 87
        quantizationTableData[3][6] = 80
        quantizationTableData[3][7] = 62
        quantizationTableData[4][0] = 18
        quantizationTableData[4][1] = 22
        quantizationTableData[4][2] = 37
        quantizationTableData[4][3] = 56
        quantizationTableData[4][4] = 68
        quantizationTableData[4][5] = 109
        quantizationTableData[4][6] = 103
        quantizationTableData[4][7] = 77
        quantizationTableData[5][0] = 24
        quantizationTableData[5][1] = 35
        quantizationTableData[5][2] = 55
        quantizationTableData[5][3] = 64
        quantizationTableData[5][4] = 81
        quantizationTableData[5][5] = 104
        quantizationTableData[5][6] = 113
        quantizationTableData[5][7] = 92
        quantizationTableData[6][0] = 49
        quantizationTableData[6][1] = 64
        quantizationTableData[6][2] = 78
        quantizationTableData[6][3] = 87
        quantizationTableData[6][4] = 103
        quantizationTableData[6][5] = 121
        quantizationTableData[6][6] = 120
        quantizationTableData[6][7] = 101
        quantizationTableData[7][0] = 72
        quantizationTableData[7][1] = 92
        quantizationTableData[7][2] = 95
        quantizationTableData[7][3] = 98
        quantizationTableData[7][4] = 112
        quantizationTableData[7][5] = 100
        quantizationTableData[7][6] = 103
        quantizationTableData[7][7] = 99
    else:
        # Standard Cb Cr channel
        # 17 18  24  47  99  99  99  99
        # 18 21  26  66  99  99  99  99
        # 24 26  56  99  99  99  99  99
        # 47 66  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99

        quantizationTableData[0][0] = 17
        quantizationTableData[0][1] = 18
        quantizationTableData[0][2] = 24
        quantizationTableData[0][3] = 47
        quantizationTableData[0][4] = 99
        quantizationTableData[0][5] = 99
        quantizationTableData[0][6] = 99
        quantizationTableData[0][7] = 99
        quantizationTableData[1][0] = 18
        quantizationTableData[1][1] = 21
        quantizationTableData[1][2] = 26
        quantizationTableData[1][3] = 66
        quantizationTableData[1][4] = 99
        quantizationTableData[1][5] = 99
        quantizationTableData[1][6] = 99
        quantizationTableData[1][7] = 99
        quantizationTableData[2][0] = 24
        quantizationTableData[2][1] = 26
        quantizationTableData[2][2] = 56
        quantizationTableData[2][3] = 99
        quantizationTableData[2][4] = 99
        quantizationTableData[2][5] = 99
        quantizationTableData[2][6] = 99
        quantizationTableData[2][7] = 99
        quantizationTableData[3][0] = 47
        quantizationTableData[3][1] = 66
        quantizationTableData[3][2] = 99
        quantizationTableData[3][3] = 99
        quantizationTableData[3][4] = 99
        quantizationTableData[3][5] = 99
        quantizationTableData[3][6] = 99
        quantizationTableData[3][7] = 99
        quantizationTableData[4][0] = 99
        quantizationTableData[4][1] = 99
        quantizationTableData[4][2] = 99
        quantizationTableData[4][3] = 99
        quantizationTableData[4][4] = 99
        quantizationTableData[4][5] = 99
        quantizationTableData[4][6] = 99
        quantizationTableData[4][7] = 99
        quantizationTableData[5][0] = 99
        quantizationTableData[5][1] = 99
        quantizationTableData[5][2] = 99
        quantizationTableData[5][3] = 99
        quantizationTableData[5][4] = 99
        quantizationTableData[5][5] = 99
        quantizationTableData[5][6] = 99
        quantizationTableData[5][7] = 99
        quantizationTableData[6][0] = 99
        quantizationTableData[6][1] = 99
        quantizationTableData[6][2] = 99
        quantizationTableData[6][3] = 99
        quantizationTableData[6][4] = 99
        quantizationTableData[6][5] = 99
        quantizationTableData[6][6] = 99
        quantizationTableData[6][7] = 99
        quantizationTableData[7][0] = 99
        quantizationTableData[7][1] = 99
        quantizationTableData[7][2] = 99
        quantizationTableData[7][3] = 99
        quantizationTableData[7][4] = 99
        quantizationTableData[7][5] = 99
        quantizationTableData[7][6] = 99
        quantizationTableData[7][7] = 99

    if QF >= 1:
        if QF < 50:
            S = 5000 / QF
        else:
            S = 200 - 2 * QF

        for i in range(8):
            for j in range(8):
                q = (50 + S * quantizationTableData[i][j]) / 100
                q = np.clip(np.floor(q), 1, 255)
                quantizationTableData[i][j] = q
    return quantizationTableData



def get_zigzag():
    zigzag = torch.tensor(( [[0,   1,   5,  6,   14,  15,  27,  28],
                             [2,   4,   7,  13,  16,  26,  29,  42],
                             [3,   8,  12,  17,  25,  30,  41,  43],
                             [9,   11, 18,  24,  31,  40,  44,  53],
                             [10,  19, 23,  32,  39,  45,  52,  54],
                             [20,  22, 33,  38,  46,  51,  55,  60],
                             [21,  34, 37,  47,  50,  56,  59,  61],
                             [35,  36, 48,  49,  57,  58,  62,  63]]))
    return zigzag

def _normalize(N: int) -> torch.Tensor:
    n = torch.ones((N, 1)).to(device)
    n[0, 0] = 1 / math.sqrt(2)
    
    return n @ n.t()

def _harmonics(N: int) -> torch.Tensor:
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)
    
def block_dct(blocks: torch.Tensor) -> torch.Tensor:
    N = blocks.shape[3]

    n = _normalize(N).float()
    h = _harmonics(N).float()

    if blocks.is_cuda:
        n = n.cuda()
        h = h.cuda()
    
    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ blocks @ h)

    return coeff

def block_idct(coeff: torch.Tensor) -> torch.Tensor:
    N = coeff.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if coeff.is_cuda:
        n = n.cuda()
        h = h.cuda()

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


class block_dct_callable(nn.Module):
    """Callable class."""
    
    def __init__(self, block_size):
        super(block_dct_callable, self).__init__()
        self.N = block_size
        self.n = _normalize(self.N).float()
        self.h = _harmonics(self.N).float()

        if torch.cuda.is_available():
            self.n = self.n.cuda()
            self.h = self.h.cuda()

    def forward(self, blocks):
        coeff = (1 / math.sqrt(2 * self.N)) * self.n * (self.h.t() @ blocks @ self.h)
        return coeff


class block_idct_callable(nn.Module):   
    def __init__(self, block_size):
        super(block_idct_callable, self).__init__()
        self.N = block_size
        self.n = _normalize(self.N).float()
        self.h = _harmonics(self.N).float()
    
        if torch.cuda.is_available():
            self.n = self.n.cuda()
            self.h = self.h.cuda()

    def forward(self, coeff):
        im = (1 / math.sqrt(2 * self.N)) * (self.h @ (self.n * coeff) @ self.h.t())
        return im

def rgb_to_ycbcr(image: torch.Tensor,
                 W_r = 0.299,
                 W_g = 0.587,
                 W_b = 0.114) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = W_r * r + W_g * g + W_b * b
    cb: torch.Tensor = (b - y) /(2*(1-W_b)) + delta
    cr: torch.Tensor = (r - y) /(2*(1-W_r)) + delta
    return torch.stack((y, cb, cr), -3)



class rgb_to_ycbcr_batch(object):
    # Define the transformation matrix as a torch tensor
    def __init__(self):
        # W_r = 0.299
        # W_g = 0.587
        # W_b = 0.114
        # self.T = torch.tensor([ [W_r, W_g, W_b],
        #                         [-W_r/2, -W_g/2, (1-W_b)/2],
        #                         [(1-W_r)/2, -W_g/2, -W_b/2]], dtype=torch.float32)

        self.T = torch.tensor([ [0.299, 0.587, 0.114],
                                [-0.168736, -0.331264, 0.5],
                                [0.5, -0.418688, -0.081312]], dtype=torch.float32)

        self.B = torch.tensor([0, 0.5, 0.5], dtype=torch.float32)
        if torch.cuda.is_available():
            self.T = self.T.cuda()
            self.B = self.B.cuda()

    def __call__(self, images: torch.Tensor)-> torch.Tensor:            
        # Reshape the batch of images from (N, 3, H, W) to (N, H*W, 3)
        N, C, H, W = images.shape
        images_reshaped = images.permute(0, 2, 3, 1).reshape(N, -1,  C)
        
        # Perform the matrix multiplication and add the bias
        ycbcr_reshaped = torch.matmul(images_reshaped, self.T.T) + self.B
        
        # Reshape back to (N, H, W, 3) and then permute to (N, 3, H, W)
        ycbcr_images = ycbcr_reshaped.view(N, H, W, C).permute(0, 3, 1, 2)
        
        return ycbcr_images

def ycbcr_to_rgb(image: torch.Tensor,
                 W_r=0.299,
                 W_g=0.587,
                 W_b=0.114) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 2*(1-W_r) * cr_shifted
    g: torch.Tensor = y - 2*(1-W_r)*W_r/W_g * cr_shifted - 2*(1-W_b)*W_b/W_g * cb_shifted
    b: torch.Tensor = y + 2*(1-W_b) * cb_shifted
    return torch.stack([r, g, b], -3)

class ycbcr_to_rgb_batch(object):
    # Define the transformation matrix as a torch tensor
    def __init__(self):
        # W_r=0.299
        # W_g=0.587
        # W_b=0.114
        # self.T_inv = torch.tensor([ [1.0, 0.0,  2*(1-W_r)], 
        #                             [1.0, - 2*(1-W_b)*W_b/W_g,  - 2*(1-W_r)*W_r/W_g], 
        #                             [1.0, 2*(1-W_b), 0.0]], dtype=torch.float32)
        
        self.T_inv = torch.tensor([[1.0, 0.0, 1.402],
                          [1.0, -0.344136, -0.714136],
                          [1.0, 1.772, 0.0]], dtype=torch.float32)
        self.B_inv = torch.tensor([0, 0.5, 0.5], dtype=torch.float32)
        
        
        if torch.cuda.is_available():
            self.T_inv = self.T_inv.cuda()
            self.B_inv = self.B_inv.cuda()

    def __call__(self, images: torch.Tensor)-> torch.Tensor:            
        # Reshape the batch of images from (N, 3, H, W) to (N, H*W, 3)
        N, C, H, W = images.shape
        images_reshaped = images.permute(0, 2, 3, 1).reshape(N, -1, C)
        
        # Subtract the bias from Cb and Cr channels
        images_reshaped -= self.B_inv
        
        # Perform the matrix multiplication
        rgb_reshaped = torch.matmul(images_reshaped, self.T_inv.T)
        
        # Reshape back to (N, H, W, 3) and then permute to (N, 3, H, W)
        rgb_images = rgb_reshaped.view(N, H, W, C).permute(0, 3, 1, 2)
        return rgb_images

def convert_NCWL_to_NWLC(img):
    return torch.transpose(torch.transpose(img,1,2),2,3)

def pad_shape(Num, size=8):
    res = Num%size
    pad = 1
    if(res == 0):
        pad = 0
    n = (Num//size+pad)*size
    return n

def blockify(im: torch.Tensor, size: int) -> torch.Tensor:
    shape = im.shape[-2:]
    padded_shape = [pad_shape(shape[0]),pad_shape(shape[1])]
    paded_im = F.pad(im, (0,padded_shape[1]-shape[1], 0,padded_shape[0]-shape[0]), 'constant',0)
    bs = paded_im.shape[0]
    ch = paded_im.shape[1]
    h = paded_im.shape[2]
    w = paded_im.shape[3]
    paded_im = paded_im.reshape(bs * ch, 1, h, w)
    paded_im = torch.nn.functional.unfold(paded_im, kernel_size=(size, size), stride=(size, size))
    paded_im = paded_im.transpose(1, 2)
    paded_im = paded_im.reshape(bs, ch, -1, size, size)
    return paded_im

def deblockify(blocks: torch.Tensor, size) -> torch.Tensor:
    padded_shape = pad_shape(size[0]),pad_shape(size[1])
    bs = blocks.shape[0]
    ch = blocks.shape[1]
    block_size = blocks.shape[3]
    blocks = blocks.reshape(bs * ch, -1, int(block_size ** 2))
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=padded_shape, kernel_size=(block_size, block_size), stride=(block_size, block_size))
    blocks = blocks.reshape(bs, ch, padded_shape[0], padded_shape[1])
    blocks = blocks[:,:,:size[0],:size[1]]
    return blocks

def load_3x3_weight(model_name = "Alexnet"):
    rt_arr = np.zeros((3,3))
    seq_weight = np.genfromtxt("color_conv_W/"+model_name+"_W_OPT.txt")
    for i in range(3):
        for j in range(3):
            rt_arr[i, j] = seq_weight[i*3+j]
    return torch.Tensor(rt_arr)


