# JPEG-Inspired-DL

This repo is for reproducing the ImageNet experimental results in our paper [*JPEG Inspired Deep Learning*] submitted at ICLR 2025.

<p align="center">
<img src="./Diagrams/JPEG-DL.png" width=80% height=80% 
class="center">
</p>

## Installation

The repo is tested with Python 3.8, CUDA 11.7, and based on the 'requirements.txt' file provided.





## Senstivity Estimation

Fetch the pretrained models used for driving the senstivity by:

    ```
    sh scripts/fetch_pretrained_cifar100.sh
    ```
which will download and save the models to `save/models`

### Senstivity Plots

<p align="center">
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Senstivity/Senstivity_Flowers/plots/densenet121.png" width="25%" height="25%" class="center">
        <figcaption>Flowers (densenet121)</figcaption>
    </figure>
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Senstivity/Senstivity_Flowers/plots/resnet18.png" width="25%" height="25%" class="center">
        <figcaption>Flowers (resnet18)</figcaption>
    </figure>
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Senstivity/Senstivity_Flowers/plots/efficientformer_l1.png" width="25%" height="25%"class="center">
        <figcaption>Flowers (efficientformer_l1)</figcaption>
    </figure>
</p>



<p align="center">
    <figure style="display: inline-block; margin: 10px;">
        <img src="./Senstivity/Senstivity_Pets/plots/densenet121.png" width="30%" height="30%" class="center">
        <figcaption>Pets (densenet121)</figcaption>
    </figure>
    <figure style="display: inline-block; margin: 10px;">
        <img src="./Senstivity/Senstivity_Pets/plots/resnet18.png" width="30%" height="30%" class="center">
        <figcaption>Pets (resnet18)</figcaption>
    </figure>
    <figure style="display: inline-block; margin: 10px;">
        <img src="./Senstivity/Senstivity_Pets/plots/efficientformer_l1.png" width="30%" height="30%" class="center">
        <figcaption>Pets (efficientformer_l1)</figcaption>
    </figure>
</p>

## Initialization of Q Tables

<p align="center">
    <img src="./Senstivity/Senstivity_Flowers/plots/densenet121.png" width="30%" height="30%" class="center">
    <img src="./Senstivity/Senstivity_Flowers/plots/resnet18.png" width="30%" height="30%" class="center">
    <img src="./Senstivity/Senstivity_Flowers/plots/efficientformer_l1.png" width="30%" height="30%" class="center">
</p>


<p align="center">
    <img src="./Senstivity/Senstivity_Pets/plots/densenet121.png" width="30%" height="30%" class="center">
    <img src="./Senstivity/Senstivity_Pets/plots/resnet18.png" width="30%" height="30%" class="center">
    <img src="./Senstivity/Senstivity_Pets/plots/efficientformer_l1.png" width="30%" height="30%" class="center">
</p>


## Robustness Testing

1. Test Robustness for the learned model 

    ```
    python3 robustness_JPEG.py --model ${mode} \
                                    --alpha_fixed --JPEG_enable \
                                    --model_dir ${model_dir}
    ```

2. Test Robustness for the standard model 

    ```
    python3 robustness_JPEG.py --model ${mode} --model_dir "./save/models/${mode}_vanilla/ckpt_epoch_240.pth
    ```



## Acknowledgements

This repo is based on the code given in [RepDistiller](https://github.com/HobbitLong/RepDistiller) for CIFAR-100 and [PyTorch](https://github.com/pytorch/vision/tree/main/references/classification#resnet) for ImageNet. Also, we use [Transformer-based](https://github.com/OscarXZQ/weight-selection.git) to produce our results for Transformer-based models. 
