teacher=resnet18
JPEG_mode=Single_Q_table
optimizer=SGD

numBits=11
hardness=0.8
JEPG_learning_rate=0.4
jpeg_layers=1

log_tag=Q_initial_abs_avg_new_norm_hardness_matching

for trail in 1 2; do
    CUDA_VISIBLE_DEVICES=2,3 python3.8 -m torch.distributed.run --master_port=25672 --nnodes=1 --nproc_per_node=2 train.py \
                                --model ${teacher} --lr 0.1 --batch-size 128 \
                                --print-freq 500 \
                                --output-dir ./save/${JPEG_mode}/${optimizer}/${teacher}_JPEG_lr_${JEPG_learning_rate}_hardness_${hardness}_numBits_${numBits}_num_layers_${jpeg_layers}_${log_tag}_${log} \
                                --JPEG_enable --JPEG_mode ${JPEG_mode} --jpeg_layers ${jpeg_layers} \
                                --optimizer ${optimizer} --JEPG_learning_rate ${JEPG_learning_rate} \
                                --hardness_matching --hardness ${hardness} \
                                --initial_Q_w_abs_avg \
                                --data-path "$SLURM_TMPDIR/imagenet"  
done
