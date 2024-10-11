teacher_list=(  
                # "vit_tiny" \
                # "mobilevitv2_050" \
                # "vit_small" \
                # "twins_pcpvt_small"
                # "mvitv2_tiny" \
                # "coatnet_pico_rw_224"
                # "coatnet_0_rw_224" \
                # "deit_tiny_patch16_224" \
                "efficientformer_l1"
			 )

dataset_list=(  
                # "CUB200" \
                # "Pets" \
                "Flowers" \
                # "STANFORD120" \
			 )

for trial in 99; do
    for teacher in ${teacher_list[*]}; do
        for dataset in ${dataset_list[*]}; do
            echo  ${dataset}
            CUDA_VISIBLE_DEVICES=6,7 python3.8 -m torch.distributed.launch --master_port=25674 --nnodes=1 --nproc_per_node=2 main.py \
                                                    --model ${teacher}  --warmup_epochs 100 --epochs 600 \
                                                    --batch_size 256 --lr 2e-3 --update_freq 1 --use_amp true \
                                                    --data_set ${dataset} --save_ckpt_freq 100 \
                                                    --output_dir ./output/Baseline/${dataset}/${teacher}_${trial} \
                                                    --data_path /home/ahamsala/PROJECT_AH/JPEG_DNN/data/
        done
    done
done


numBits=8
trial=100

# --zero_dc

for dataset in ${dataset_list[*]}; do
    for teacher in ${teacher_list[*]}; do
        for q_max in 5; do
            for JEPG_alpha in 5; do
                start_1=0.005
                end_1=0.005
                step_1=0.001
                while (( $(awk -v start="$start_1" -v end="$end_1" 'BEGIN { if (start <= end) print 1; else print 0 }') )); do
                    JEPG_learning_rate=${start_1}
                    echo "JEPG_learning_rate: ${JEPG_learning_rate}"

                    # CUDA_VISIBLE_DEVICES=6,7 python3.8 -m torch.distributed.launch --master_port=25674 --nnodes=1 --nproc_per_node=2 main_JPEG.py \
                    #                                         --model ${teacher}  --warmup_epochs 100 --epochs 600 \
                    #                                         --batch_size 256 --lr 2e-3 --update_freq 1 --use_amp true \
                    #                                         --data_set ${dataset} --save_ckpt_freq 100 \
                    #                                         --output_dir ./save/JPEG/${dataset}/${teacher}_zeroDC_${trial} \
                    #                                         --data_path /home/ahamsala/PROJECT_AH/JPEG_DNN/data/ \
                    #                                         --zero_dc --JPEG_enable

                    # CUDA_VISIBLE_DEVICES=0,1 python3.8 -m torch.distributed.launch --master_port=25670 --nnodes=1 --nproc_per_node=2 main_JPEG.py \
                    #                                         --model ${teacher}  --warmup_epochs 100 --epochs 600 \
                    #                                         --batch_size 256 --lr 2e-3 --update_freq 1 --use_amp true \
                    #                                         --data_set ${dataset} --save_ckpt_freq 100 \
                    #                                         --output_dir ./save/JPEG/${dataset}/${teacher}_JPEG_lr_${JEPG_learning_rate}_alpha_${JEPG_alpha}_q_max_${q_max}_numBits_${numBits}_${trial} \
                    #                                         --data_path /home/ahamsala/PROJECT_AH/JPEG_DNN/data/ \
                    #                                         --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                    #                                         --JEPG_alpha ${JEPG_alpha} \
                    #                                         --alpha_fixed --q_max ${q_max} \
                    #                                         --initial_Q_w_sensitivity 
                    start_1=$(awk -v start="$start_1" -v step="$step_1" 'BEGIN { printf "%.5f", start + step }')
                done
            done
        done
    done
done