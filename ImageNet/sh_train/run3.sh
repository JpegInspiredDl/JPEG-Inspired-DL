teacher=squeezenet1_1
JPEG_mode=Single_Q_table
optimizer=SGD

numBits=11
hardness=3.0

start_1=0.1
end_1=0.1
step_1=0.1

log_tag=Q_initial_abs_avg_new_norm_hardness_matching

while (( $(awk -v start="$start_1" -v end="$end_1" 'BEGIN { if (start <= end) print 1; else print 0 }') )); do
    JEPG_learning_rate=${start_1}
    echo "JEPG_learning_rate: ${JEPG_learning_rate}"
    CUDA_VISIBLE_DEVICES=6,7 python3.8 -m torch.distributed.run --master_port=25672 --nnodes=1 --nproc_per_node=2 train.py \
                                --model ${teacher} --epochs 100 --lr 0.01 --weight-decay 0.0002 --batch-size 64 \
                                --print-freq 500 \
                                --output-dir ./save/save/${JPEG_mode}/${optimizer}/${teacher}_JPEG_lr_${JEPG_learning_rate}_hardness_${hardness}_numBits_${numBits}_${log_tag} \
                                --JPEG_enable --JPEG_mode ${JPEG_mode}  \
                                --optimizer ${optimizer} --JEPG_learning_rate ${JEPG_learning_rate} \
                                --hardness_matching --hardness ${hardness} \
                                --initial_Q_w_abs_avg \
                                --data-path "$SLURM_TMPDIR/imagenet"  
    start_1=$(awk -v start="$start_1" -v step="$step_1" 'BEGIN { printf "%.5f", start + step }')
done