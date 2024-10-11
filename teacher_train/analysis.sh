GPU_ID=0


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


trial=1
CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 std_analysis.py --model "resnet18" -t ${trial} \
        --dataset "ImageNet" --batch_size 256 --JPEG_enable


# hardness=0.5

# min_Q_Step=0

# std_width_Y=2
# std_width_CbCr=2

# start_1=3
# end_1=3
# step_1=0.5


# log_add=_clampWbits_Init_Q

# for trial in 1; do
#     for teacher in ${teacher_list[*]}; do
#         for hardness in $(seq $start_1 $step_1 $end_1); do
#             echo "JEPG_learning_rate: ${JEPG_learning_rate}"
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar.py --model ${teacher} -t ${trial} \
#                                             --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
#                                             --hardness ${hardness} --min_Q_Step ${min_Q_Step}\
#                                             --log_add ${log_add} --seed 11
#         done

#     done
# done