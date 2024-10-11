from torchvision import models

def get_model(Model):
    transform = None
    if Model=="Resnet18":
        pretrained_model = models.resnet18(pretrained=True).eval()
    elif Model=="Resnet34":
        pretrained_model = models.resnet34(pretrained=True).eval()
    elif Model=="Resnet50":
        pretrained_model = models.resnet50(pretrained=True).eval()
    elif Model=="Resnet101":
        pretrained_model = models.resnet101(pretrained=True).eval()
    elif Model=="Resnet152":
        pretrained_model = models.resnet152(pretrained=True).eval()
    
    elif Model=="Squeezenet":
        pretrained_model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT).eval()

    elif Model == 'Shufflenetv2_05':
        pretrained_model = models.shufflenet_v2_x0_5(pretrained=True).eval()
    elif Model == 'Shufflenetv2_10':
        pretrained_model = models.shufflenet_v2_x1_0(pretrained=True).eval()
    elif Model == 'Shufflenetv2_15':
        pretrained_model = models.shufflenet_v2_x1_5(pretrained=True).eval()
    elif Model == 'Shufflenetv2_20':
        pretrained_model = models.shufflenet_v2_x2_0(pretrained=True).eval()

    elif Model == 'ViT_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        # transform = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms
        pretrained_model = models.vit_b_16(pretrained=True).eval()
    elif Model == 'ViT_b_32':
        weights = models.ViT_B_32_Weights.IMAGENET1K_V1
        # transform = models.ViT_B_32_Weights.IMAGENET1K_V1.transforms
        pretrained_model = models.vit_b_32(pretrained=True).eval()
    elif Model == 'ViT_l_16':
        weights = models.ViT_L_16_Weights.IMAGENET1K_V1
        # transform = models.ViT_L_16_Weights.IMAGENET1K_V1.transforms
        pretrained_model = models.vit_l_16(pretrained=True).eval()
    elif Model == 'ViT_l_32':
        weights = models.ViT_L_32_Weights.IMAGENET1K_V1
        pretrained_model = models.vit_l_32(pretrained=True).eval()

    elif Model == 'MaxVit_T':
        weights = models.MaxVit_T_Weights.IMAGENET1K_V1
        pretrained_model = models.maxvit_t(pretrained=True).eval()

    elif Model == 'Swin_T':
        weights = models.Swin_T_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_t(pretrained=True).eval()
    elif Model == 'Swin_S':
        weights = models.Swin_S_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_s(pretrained=True).eval()
    elif Model == 'Swin_B':
        weights = models.Swin_B_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_b(pretrained=True).eval()

    elif Model == 'Swin_V2_T':
        weights = models.Swin_V2_T_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_v2_t(pretrained=True).eval()
    elif Model == 'Swin_V2_S':
        weights = models.Swin_V2_S_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_v2_s(pretrained=True).eval()
    elif Model == 'Swin_V2_B':
        weights = models.Swin_V2_B_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_v2_b(pretrained=True).eval()

    elif Model == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        pretrained_model = models.mobilenet_v3_small(weights=weights).eval()

    elif Model == 'mobilenet_v2_RV2':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
        pretrained_model = models.mobilenet_v2(weights=weights).eval()
    elif Model == 'mobilenet_v2':
        weights = models.MobileNet_V2_Weights.DEFAULT
        pretrained_model = models.mobilenet_v2(weights=weights).eval()
    
    # mnasnet0_5
    elif Model == 'mnasnet1_0':
        pretrained_model = models.mnasnet1_0(pretrained=True).eval()
    elif Model == 'mnasnet0_5':
        pretrained_model = models.mnasnet0_5(pretrained=True).eval()

    elif Model == 'Alexnet':
        weights= models.AlexNet_Weights.DEFAULT
        pretrained_model = models.alexnet(weights=weights).eval()
    
    elif Model == 'ConvNeXt_base':
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        pretrained_model = models.convnext_base(weights=weights).eval()
    elif Model == 'ConvNeXt_tiny':
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        pretrained_model = models.convnext_tiny(weights=weights).eval()
    elif Model == 'ConvNeXt_large':
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        pretrained_model = models.convnext_large(weights=weights).eval()
    elif  Model == 'ConvNeXt_small':
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        pretrained_model = models.convnext_small(weights=weights).eval()


    elif Model == 'WRNs101_V1':
        weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
        pretrained_model = models.wide_resnet101_2(weights=weights).eval()
    elif Model == 'WRNs101_V2':
        weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V2
        pretrained_model = models.wide_resnet101_2(weights=weights).eval()
    elif Model == 'WRNs50_V1':
        weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        pretrained_model = models.wide_resnet50_2(weights=weights).eval()
    elif Model == 'WRNs50_V2':
        weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
        pretrained_model = models.wide_resnet50_2(weights=weights).eval()


    elif Model == 'DenseNet121':
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
        pretrained_model = models.densenet121(weights=weights).eval()
    elif Model == 'DenseNet161':
        weights=models.DenseNet161_Weights.IMAGENET1K_V1
        pretrained_model = models.densenet161(weights=weights).eval()
    elif Model == 'DenseNet169':
        weights=models.DenseNet169_Weights.IMAGENET1K_V1
        pretrained_model = models.densenet169(weights=weights).eval()
    elif Model == 'DenseNet201':
        weights=models.DenseNet201_Weights.IMAGENET1K_V1
        pretrained_model = models.densenet201(weights=weights).eval()

    elif Model == 'EfficientNet_B0':
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        pretrained_model = models.efficientnet_b0(weights=weights).eval()
    
    elif Model == 'Regnet400mf':
        pretrained_model = models.regnet_y_400mf(pretrained=True).eval()
    elif Model == 'Regnet400mf_V2':
        weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2
        pretrained_model = models.regnet_y_400mf(weights=weights).eval()

    elif Model == 'Regnet800mf':
        pretrained_model = models.regnet_y_800mf(pretrained=True).eval()
    elif Model == 'Regnet800mf_V2':
        weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2
        pretrained_model = models.regnet_y_800mf(weights=weights).eval()

    elif Model == 'Regnet6gf':
        pretrained_model = models.regnet_y_1_6gf(pretrained=True).eval()
    
    elif Model == 'Regnet2gf':
        pretrained_model = models.regnet_y_3_2gf(pretrained=True).eval()
    elif Model == 'Regnet2gf_V2':
        weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2
        pretrained_model = models.regnet_y_3_2gf(weights=weights).eval()

    elif Model == 'Regnet8gf':
        pretrained_model = models.regnet_y_8gf(pretrained=True).eval()
    elif Model == 'Regnet8gf_V2':
        weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V2
        pretrained_model = models.regnet_y_8gf(weights=weights).eval()

    elif Model == 'Regnet16gf':
        pretrained_model = models.regnet_y_16gf(pretrained=True).eval()
    elif Model == 'Regnet32gf':
        pretrained_model = models.regnet_y_32gf(pretrained=True).eval()


    return pretrained_model

