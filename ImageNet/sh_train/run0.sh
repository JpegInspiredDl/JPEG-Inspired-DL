teacher=mobilenet_v2
JPEG_mode=Single_Q_table
optimizer=SGD

numBits=11
hardness=0.7
jpeg_layers=10

JEPG_learning_rate=0.5
jpeg_layers=1

log_tag=Q_initial_abs_avg_new_norm_hardness_matching

CUDA_VISIBLE_DEVICES=4,5,6,7 python3.8 -m torch.distributed.run --master_port=25671 --nnodes=1 --nproc_per_node=4 train.py \
                            --model ${teacher} --epochs 300 --lr 0.045 --wd 0.00004  --batch-size 64 \
                            --lr-step-size 1 --lr-gamma 0.98\
                            --print-freq 500 \
                            --output-dir ./save/${JPEG_mode}/${optimizer}/${teacher}_JPEG_lr_${JEPG_learning_rate}_hardness_${hardness}_numBits_${numBits}_num_layers_${jpeg_layers}_${log_tag} \
                            --JPEG_enable --JPEG_mode ${JPEG_mode} --jpeg_layers ${jpeg_layers} \
                            --optimizer ${optimizer} --JEPG_learning_rate ${JEPG_learning_rate} \
                            --hardness_matching --hardness ${hardness} \
                            --initial_Q_w_abs_avg \
                            --data-path "$SLURM_TMPDIR/imagenet"  
