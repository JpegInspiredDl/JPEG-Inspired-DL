teacher_list=(  
                # CUB 200
                # "resnet18" \ 
				"vgg8" \
                # "resnet20" \
				# "wrn_40_1" \
                # "vgg13" \
				# "MobileNetV2"\
                # "wrn_40_2" \
                # "resnet32" \
                # "ShuffleV1" \
                # "ShuffleV2" \
				)


hardness=2.4

min_Q_Step=0
max_Q_Step=255
num_bit=8

start_1=0.7
end_1=0.9
step_1=0.1

log_add=_clampWbits_Init_Q


for teacher in ${teacher_list[*]}; do
    while (( $(awk -v start="$start_1" -v end="$end_1" 'BEGIN { if (start <= end) print 1; else print 0 }') )); do
        JEPG_learning_rate=${start_1}
        echo "JEPG_learning_rate: $JEPG_learning_rate"
        for trial in 1 2; do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar.py --model ${teacher} -t ${trial} \
                                            --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                                            --hardness ${hardness} \
                                            --min_Q_Step ${min_Q_Step} --max_Q_Step ${max_Q_Step} \
                                            --log_add ${log_add}  --num_bit ${num_bit}
        done
        start_1=$(awk -v start="$start_1" -v step="$step_1" 'BEGIN { printf "%.5f", start + step }')
    done
done


# --------------------------------------


teacher_list=(  
                # CUB 200
                # "resnet18" \ 
				# "vgg8" \
                "vgg13" \
                # "resnet20" \
				# "wrn_40_1" \
				# "MobileNetV2"\
                # "wrn_40_2" \
                # "resnet32" \
                # "ShuffleV1" \
                # "ShuffleV2" \
				)


parameters_list=(  
				# "--JEPG_learning_rate 0.4 --hardness 0.4 --num_bit 11" \
                # "--JEPG_learning_rate 0.5 --hardness 0.5 --num_bit 11" \
                # "--JEPG_learning_rate 0.6 --hardness 0.6 --num_bit 11" \
                # "--JEPG_learning_rate 0.4 --hardness 0.5 --num_bit 8" \
                "--JEPG_learning_rate 0.4 --hardness 0.7 --num_bit 8" \
                "--JEPG_learning_rate 0.4 --hardness 0.8 --num_bit 8" \
                "--JEPG_learning_rate 0.4 --hardness 0.9 --num_bit 8" \
                # "--JEPG_learning_rate 0.4 --hardness 1.0 --num_bit 8" \
                # "--JEPG_learning_rate 0.5 --hardness 0.6 --num_bit 8" \
                # "--JEPG_learning_rate 0.5 --hardness 0.7 --num_bit 8" \
                # "--JEPG_learning_rate 0.6 --hardness 0.5 --num_bit 8" \
                # "--JEPG_learning_rate 0.6 --hardness 0.8 --num_bit 8" \
                # "--JEPG_learning_rate 0.6 --hardness 1.1 --num_bit 8" \
                # "--JEPG_learning_rate 0.6 --hardness 1.2 --num_bit 8" \
                # "--JEPG_learning_rate 0.7 --hardness 1.6 --num_bit 8" \
                # "--JEPG_learning_rate 0.7 --hardness 2.0 --num_bit 8" \
                # "--JEPG_learning_rate 0.8 --hardness 1.0 --num_bit 8" \
                # "--JEPG_learning_rate 0.8 --hardness 1.8 --num_bit 8" \
                # "--JEPG_learning_rate 0.9 --hardness 1.0 --num_bit 8" \
                # "--JEPG_learning_rate 0.9 --hardness 2.2 --num_bit 8" \
				)


for teacher in ${teacher_list[*]}; do
    for parameters in "${parameters_list[@]}"; do
        for trial in 1 2; do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar.py --model ${teacher} -t ${trial} \
                                            --JPEG_enable ${parameters} 
        done
    done
done

# --------------------------------------

method_list=(  
                # "--alpha_fixed --ADAM_enable" \
                # "--alpha_fixed --initial_Q_abs_avg" \
                # "--alpha_fixed --initial_Q_w_sensitivity --JPEG_alpha_trainable --ADAM_enable" \
                "--JPEG_alpha_trainable --initial_Q_w_sensitivity --ADAM_enable" \
                #  "--alpha_fixed --initial_Q_w_sensitivity" \
                # "--alpha_scaling" \
                # "--alpha_scaling --initial_Q_abs_avg" \
                # "--alpha_scaling --initial_Q_w_sensitivity" \
				)

JEPG_alpha=0.7

start_1=0.001
end_1=0.01
step_1=0.002

for teacher in ${teacher_list[*]}; do
    for method in "${method_list[@]}"; do
        while (( $(awk -v start="$start_1" -v end="$end_1" 'BEGIN { if (start <= end) print 1; else print 0 }') )); do
            JEPG_learning_rate=${start_1}
            echo "JEPG_learning_rate: $JEPG_learning_rate"
                for trial in 1 2; do
                    time CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar_JPEG.py --model ${teacher} -t ${trial} \
                                                    --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                                                    --JEPG_alpha ${JEPG_alpha} ${method}
                done
            start_1=$(awk -v start="$start_1" -v step="$step_1" 'BEGIN { printf "%.5f", start + step }')
        done
    done
done

# --------------------------------------


teacher_list=(  
                # CUB 200
                # "resnet18" \ 
				# "--model vgg8 " \
                # "--model vgg8 " \
                # "vgg8" \
                # "resnet20" \
                "resnet56" \
				# "wrn_40_1" \
                # "vgg13" \
				# "MobileNetV2"\
                # "wrn_40_2" \
                # "resnet32" \
                # "ShuffleV1" \
                # "ShuffleV2" \
				)

JEPG_alpha=5

start_1=0.001
end_1=0.006
step_1=0.002

for teacher in "${teacher_list[*]}"; do
    while (( $(awk -v start="$start_1" -v end="$end_1" 'BEGIN { if (start <= end) print 1; else print 0 }') )); do
        JEPG_learning_rate=${start_1}
        echo "JEPG_learning_rate: $JEPG_learning_rate method: ${method}"
        for q_max in 5 10 20; do
            for trial in 1 2; do
                time CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar_JPEG.py --model ${teacher} -t ${trial} \
                                                --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                                                --JEPG_alpha ${JEPG_alpha} \
                                                --alpha_fixed --initial_Q_w_sensitivity --ADAM_enable \
                                                --q_max ${q_max}
            done
        done
        start_1=$(awk -v start="$start_1" -v step="$step_1" 'BEGIN { printf "%.5f", start + step }')
    done
done