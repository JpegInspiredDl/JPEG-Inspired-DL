teacher_list=(  
                # "vit_tiny" \ 
                # "mobilevitv2_050" \
                # "vit_small" \
                # "twins_pcpvt_small"
                # "deit_tiny_patch16_224"
                # "efficientformer_l1"
                "efficientformer_l1"
			 )

trial=99

dataset_list=(  
                "CUB200" \
                "Pets" \
                # "Flowers" \
                "STANFORD120" \
			 )


for dataset in ${dataset_list[*]}; do
    for teacher in ${teacher_list[*]}; do
        CUDA_VISIBLE_DEVICES=0 python3.8 vit_sensitivity_calc.py \
                                        --model ${teacher}  --batch_size 64 \
                                        --data_set ${dataset} \
                                        --output_dir ./output/Baseline/${dataset}/${teacher}_${trial} \
                                        --data_path /home/ahamsala/PROJECT_AH/JPEG_DNN/data/
    done
done

