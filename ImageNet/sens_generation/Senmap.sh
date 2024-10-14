GPU_ID=5

# ConvNeXt_tiny Regnet400mf EfficientNet_B0 mobilenet_v2 Mnasnet

# for model in Regnet400mf EfficientNet_B0 mobilenet_v2 Mnasnet 
# MaxVit_T Swin_T Swin_S Swin_B Swin_V2_T Swin_V2_S Swin_V2_B 
# Regnet400mf_V2
for model in Squeezenet
do
    # CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 get_DCTgrad.py -model ${model} -Batch_size 10 -Nexample 10000
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 get_DCtgrad_Multi_Q.py -model ${model} -Batch_size 10 -Nexample 10000
done




