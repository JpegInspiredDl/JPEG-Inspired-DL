GPU_ID=5

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


JEPG_alpha=0.8

start_1=0.04
end_1=0.1
step_1=0.02

for teacher in ${teacher_list[*]}; do
    while (( $(awk -v start="$start_1" -v end="$end_1" 'BEGIN { if (start <= end) print 1; else print 0 }') )); do
        JEPG_learning_rate=${start_1}
        echo "JEPG_learning_rate: $JEPG_learning_rate"
        for trial in 1 2; do
            time CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar.py --model ${teacher} -t ${trial} \
                                            --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                                            --JEPG_alpha ${JEPG_alpha} --alpha_scaling
        done
        start_1=$(awk -v start="$start_1" -v step="$step_1" 'BEGIN { printf "%.5f", start + step }')
    done
done