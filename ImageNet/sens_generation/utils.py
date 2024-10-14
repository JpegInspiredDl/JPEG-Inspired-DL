import torch
import torch.nn.functional as F
import math
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

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
    N = blocks.shape[-1]

    n = _normalize(N).float()
    h = _harmonics(N).float()

    if blocks.is_cuda:
        n = n.cuda()
        h = h.cuda()
    
    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ blocks @ h)

    return coeff

def block_idct(coeff: torch.Tensor) -> torch.Tensor:
    N = coeff.shape[-1]

    n = _normalize(N)
    h = _harmonics(N)

    if coeff.is_cuda:
        n = n.cuda()
        h = h.cuda()

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
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


def plot_confidence_interval(x, top, bottom, mean, horizontal_line_width=0.25, color='#2187bb',label=None,alpha=1):
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2
    plt.plot([x, x], [top, bottom], color=color,alpha=0.7*alpha)
    plt.plot([left, right], [top, top], color=color,alpha=0.7*alpha)
    plt.plot([left, right], [bottom, bottom], color=color,alpha=0.7*alpha)
    plt.plot(x, mean, 'o', color=color, label=label,alpha=alpha)
    return mean