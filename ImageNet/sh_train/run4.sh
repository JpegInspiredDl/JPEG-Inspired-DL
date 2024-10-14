teacher_list=(  
                "resnet18" \ 
				)


numBits=11
q_max=10

for teacher in ${teacher_list[*]}; do
    for JEPG_alpha in 10; do
        start_1=0.001
        end_1=0.003
        step_1=0.002
        while (( $(awk -v start="$start_1" -v end="$end_1" 'BEGIN { if (start <= end) print 1; else print 0 }') )); do
            JEPG_learning_rate=${start_1}
            echo "JEPG_learning_rate: ${JEPG_learning_rate}"
            CUDA_VISIBLE_DEVICES=2,3 python3.8 -m torch.distributed.run --master_port=25674 --nnodes=1 --nproc_per_node=2 train.py \
                                        --model ${teacher} --batch-size 128  \
                                        --print-freq 500 --JPEG_enable \
                                        --output-dir ./save/${teacher}_JPEG_lr_${JEPG_learning_rate}_alpha_${JEPG_alpha}_q_max_${q_max}_numBits_${numBits} \
                                        --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                                        --JEPG_alpha ${JEPG_alpha} \
                                        --alpha_fixed --initial_Q_w_sensitivity  --q_max ${q_max}
            start_1=$(awk -v start="$start_1" -v step="$step_1" 'BEGIN { printf "%.5f", start + step }')
        done
    done
done