GPU_ID=4


dataset_list=(  
                # CUB 200
                # "CUB200"  \ 
                # "STANFORD120" \
                "Pets" \
                # "Flowers"
				)


teacher_list=(  
                # CUB 200
                # "resnet18" \ 
                "densenet121" \
				)

for dataset in ${dataset_list[*]}; do
    for teacher in ${teacher_list[*]}; do
        # time CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher.py --model ${teacher} \
        #         --dataset ${dataset} 

        time CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher.py --model ${teacher} \
                --dataset ${dataset} \
                --find_sensitivity
    done
done