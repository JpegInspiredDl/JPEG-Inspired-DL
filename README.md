# JPEG-Inspired-DL

This repo is for reproducing the ImageNet experimental results in our paper [*JPEG Inspired Deep Learning*] submitted at ICLR 2025.

<p align="center">
<img src="./scripts/JPEG-DL.png" width=80% height=80% 
class="center">
</p>

## Installation

The repo is tested with Python 3.8, PyTorch 2.0.1, and CUDA 11.7.


## Running

1. Fetch the pretrained models used for driving the senstivity by:
    ```
    sh scripts/fetch_pretrained_cifar100.sh
    ```
   which will download and save the models to `save/models`


## Acknowledgements

This repo is based on the code given in [RepDistiller](https://github.com/HobbitLong/RepDistiller) for CIFAR-100 and [PyTorch](https://github.com/pytorch/vision/tree/main/references/classification#resnet) for ImageNet. Also, we use [Transformer-based](https://github.com/yoshitomo-matsubara/torchdistill) to produce our results for Transformer-based models. 
