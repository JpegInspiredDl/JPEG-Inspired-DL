teacher_list=(  
#                 # "vit_small" \
#                 "vit_tiny" \
#                 # "mobilevitv2_050" \
#                 # "twins_pcpvt_small"
#                 # "mvitv2_tiny" \
#                 # "coatnet_pico_rw_224"
#                 # "coatnet_0_rw_224" \
#                 # "deit_tiny_patch16_224" \
                  "efficientformer_l1"
			 )


dataset=cifar100
# for teacher in ${teacher_list[*]}; do
#     for trial in 10 11 12 13; do
#         CUDA_VISIBLE_DEVICES=2,3 python3.8 -m torch.distributed.launch --master_port=25665 --nnodes=1 --nproc_per_node=2 main.py \
#                                                 --model ${teacher}  --warmup_epochs 50 --epochs 300 \
#                                                 --batch_size 256 --lr 2e-3 --update_freq 1 --use_amp true \
#                                                 --data_set ${dataset} --save_ckpt_freq 100 \
#                                                 --output_dir ./output/Baseline/${dataset}/${teacher}_${trial} 
#     done
# done


numBits=8
alpha=7
JEPG_learning_rate=0.003
trial=1

log=initial_ones_fixed_alpha_adam

for teacher in ${teacher_list[*]}; do
        for trial in 3 4; do
                CUDA_VISIBLE_DEVICES=6,7 python3.8 -m torch.distributed.launch --master_port=25674 --nnodes=1 --nproc_per_node=2 main_JPEG.py \
                                                        --model ${teacher}  --warmup_epochs 50 --epochs 300 \
                                                        --batch_size 256 --lr 2e-3 --update_freq 1 --use_amp true \
                                                        --data_set ${dataset} --save_ckpt_freq 100 \
                                                        --output_dir ./save/JPEG/${teacher}/${teacher}_JPEG_lr_${JEPG_learning_rate}_alpha_${alpha}_numBits_${numBits}_${log}_${trial} \
                                                        --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                                                        --alpha_fixed --JEPG_alpha ${alpha} 
        done                                            
done