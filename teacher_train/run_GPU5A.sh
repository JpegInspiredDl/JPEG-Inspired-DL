GPU_ID=5

parameters_list=(  
                # "--model vgg13 --JEPG_learning_rate 0.001 --JEPG_alpha 5.0 --q_max 5.0" \
                # "--model vgg13 --JEPG_learning_rate 0.003 --JEPG_alpha 1.0 --q_max 10.0" \
                # "--model vgg13 --JEPG_learning_rate 0.003 --JEPG_alpha 5.0 --q_max 10.0" \
                # "--model vgg13 --JEPG_learning_rate 0.003 --JEPG_alpha 1.0 --q_max 20.0" \
                # "--model resnet56 --JEPG_learning_rate 0.003 --JEPG_alpha 1.0 --q_max 5.0" \
                # "--model resnet56 --JEPG_learning_rate 0.001 --JEPG_alpha 5.0 --q_max 10.0" \
                # "--model resnet56 --JEPG_learning_rate 0.003 --JEPG_alpha 1.0 --q_max 10.0" \
                # "--model resnet56 --JEPG_learning_rate 0.004 --JEPG_alpha 1.0 --q_max 10.0" \
                # "--model resnet56 --JEPG_learning_rate 0.005 --JEPG_alpha 1.0 --q_max 10.0" \
                # "--model resnet56 --JEPG_learning_rate 0.005 --JEPG_alpha 1.0 --q_max 20.0" \
                "--model resnet32 --JEPG_learning_rate 0.003 --JEPG_alpha 1.0 --q_max 10.0" \
                # "--model resnet32 --JEPG_learning_rate 0.004 --JEPG_alpha 1.0 --q_max 10.0" \
                # "--model resnet32 --JEPG_learning_rate 0.003 --JEPG_alpha 1.0 --q_max 15.0" \
                # "--model resnet32 --JEPG_learning_rate 0.004 --JEPG_alpha 1.0 --q_max 15.0" \
				)

for parameters in "${parameters_list[@]}"; do
    for trial in 10 11 12 13 14 15; do
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar_JPEG.py -t ${trial} \
                                        --JPEG_enable ${parameters} --alpha_fixed --initial_Q_w_sensitivity --ADAM_enable
    done
done