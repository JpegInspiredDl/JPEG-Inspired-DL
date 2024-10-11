GPU_ID=0

teacher_list=(  
                # CUB 200
                # "resnet18" \ 
				# "--model vgg8 " \
                # "--model vgg8 " \
                # "vgg8" \
                # "resnet20" \
				# "wrn_40_1" \
                "vgg13" \
				# "MobileNetV2"\
                # "wrn_40_2" \
                # "resnet32" \
                # "ShuffleV1" \
                # "ShuffleV2" \
				)

method_list=(  
                "--alpha_fixed --ADAM_enable" \
                # "--alpha_fixed --initial_Q_abs_avg" \
                # "--alpha_fixed --initial_Q_w_sensitivity" \
                # "--alpha_scaling" \
                # "--alpha_scaling --initial_Q_abs_avg" \
                # "--alpha_scaling --initial_Q_w_sensitivity" \
				)

JEPG_alpha=1

for teacher in "${teacher_list[*]}"; do
    for JEPG_learning_rate in 0.0001 0.0005 0.001 0.005; do
        for method in "${method_list[@]}"; do
            echo "JEPG_learning_rate: $JEPG_learning_rate method: ${method}"
            for trial in 1 2; do
                time CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar_JPEG.py --model ${teacher} -t ${trial} \
                                                --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                                                --JEPG_alpha ${JEPG_alpha} ${method}
            done
        done
    done
done