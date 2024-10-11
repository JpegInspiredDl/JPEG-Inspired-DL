import numpy as np
import torch
import torch.nn as nn

import scipy.fftpack as fftpack

EPS = 1e-10



try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))
    
    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)
    
    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)
    

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)



def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

class DCTLayer(nn.Module):
    def __init__(self):
        super(DCTLayer, self).__init__()

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()

        # Ensure input dimensions are divisible by 8 for simplicity
        new_height = height + (8 - height % 8) if height % 8 != 0 else height
        new_width = width + (8 - width % 8) if width % 8 != 0 else width
        inputs = torch.nn.functional.pad(inputs, (0, new_width - width, 0, new_height - height))

        # Reshape input into 8x8 blocks
        inputs = inputs.unfold(2, 8, 8).unfold(3, 8, 8)
        inputs = inputs.contiguous().view(batch_size, channels, -1, 8, 8)

        breakpoint()

        # Forward DCT
        inputs_dct = dct(inputs)

        # Inverse DCT
        reconstructed_blocks = idct(inputs_dct)

        # Concatenate reconstructed blocks into the original image shape
        reconstructed_image = reconstructed_blocks.view(batch_size, channels, -1, new_height, new_width)
        return reconstructed_image


def test_dct_2d():
    for N1 in [2, 5, 32]:
        for N2 in [2, 5, 32]:
            x = np.random.normal(size=(1, N1, N2))
            ref = fftpack.dct(x, axis=2, type=2)
            ref = fftpack.dct(ref, axis=1, type=2)
            act = dct_2d(torch.tensor(x)).numpy()
            assert np.abs(ref - act).max() < EPS, (ref, act)


def test_idct_2d():
    for N1 in [2, 5, 32]:
        for N2 in [2, 5, 32]:
            x = np.random.normal(size=(1, N1, N2))
            X = dct_2d(torch.tensor(x))
            y = idct_2d(X).numpy()
            assert np.abs(x - y).max() < EPS, x


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)


# # Example usage
# input_image = torch.rand(1, 3, 224, 224)  # Example 256x256 RGB image
# dct_layer = DCTLayer()
# reconstructed_image = dct_layer(input_image)
# print("Reconstructed image shape:", reconstructed_image.shape)

breakpoint()


x = torch.randn((200, 200))
X = dct(x)   # DCT-II done through the last dimension
y = idct(X)  # scaled DCT-III done through the last dimension
print( (torch.abs(x - y)).sum())   # x == y within numerical t

# x = torch.Tensor(1000,4096)
# x.normal_(0,1)
# linear_dct = LinearDCT(4096, 'dct')
# error = torch.abs(dct(x) - linear_dct(x))
# assert error.max() < 1e-3, (error, error.max())
# linear_idct = LinearDCT(4096, 'idct')
# error = torch.abs(idct(x) - linear_idct(x))
# assert error.max() < 1e-3, (error, error.max())