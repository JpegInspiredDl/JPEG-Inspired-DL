GPU_ID=4
CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 robustness_JPEG.py --model resnet56 \
                                --dataset cifar100 \
                                --alpha_fixed --JPEG_enable
