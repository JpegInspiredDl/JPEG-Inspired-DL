


teacher=resnet18
JPEG_mode=Single_Q_table
optimizer=ADAM

dir=/home/ahamsala/PROJECTS/JPEG_DNN/imagenet_torch/hardness_matching/resnet18_JPEG_lr_0.5_hardness_1_numBits_11_Q_min_0_Q_max_255.0/

CUDA_VISIBLE_DEVICES=6 python3.8 val.py \
                            --model ${teacher} --batch-size 128 --lr 0.1 \
                            --JPEG_enable --JPEG_mode ${JPEG_mode} --test-only\
                            --optimizer ${optimizer} \
                            --resume ${dir}model_best.pth \
                            --output-dir ${dir}

