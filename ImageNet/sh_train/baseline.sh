teacher_list=(  
                "resnet18" \ 
				)

for teacher in ${teacher_list[*]}; do
    CUDA_VISIBLE_DEVICES=1 python3.8 -m torch.distributed.run --master_port=25680 --nnodes=1 --nproc_per_node=1 main.py \
                    --model ${teacher} --batch-size 256  \
                    --print-freq 500 \
                    --output-dir ./save/baseline/${teacher}
done
