
teacher_list=(  
                "shufflenet_v2_x0_5" \ 
				)


numBits=11
q_max=5

for teacher in ${teacher_list[*]}; do
    for JEPG_alpha in 5; do
        start_1=0.001
        end_1=0.003
        step_1=0.002
        while (( $(awk -v start="$start_1" -v end="$end_1" 'BEGIN { if (start <= end) print 1; else print 0 }') )); do
            JEPG_learning_rate=${start_1}
            echo "JEPG_learning_rate: ${JEPG_learning_rate}"
            CUDA_VISIBLE_DEVICES=4 python3.8 -m torch.distributed.run --master_port=25677 --nnodes=1 --nproc_per_node=1 train.py \
                                        --model ${teacher} \
                                        --print-freq 500 \
                                        --batch-size 512 \
                                        --lr 0.5 --lr-scheduler=cosineannealinglr --lr-warmup-epochs=5 --lr-warmup-method linear \
                                        --auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
                                        --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
                                        --train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps 4 \
                                        # --output-dir ./save/${teacher}_JPEG_lr_${JEPG_learning_rate}_alpha_${JEPG_alpha}_q_max_${q_max}_numBits_${numBits} \
                                        # --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                                        # --JEPG_alpha ${JEPG_alpha} \
                                        # --alpha_fixed --initial_Q_w_sensitivity  --q_max ${q_max}
            start_1=$(awk -v start="$start_1" -v step="$step_1" 'BEGIN { printf "%.5f", start + step }')
        done
    done
done