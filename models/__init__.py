from .resnet import *
from .densenet import *
from .densenet3 import *
from torchvision import models

def load_model(name, num_classes=10, pretrained=False, **kwargs):
    model_dict = globals()
    model = model_dict[name](pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model



# def load_model_imagenet(name, pretrained=False, **kwargs):
#     if name=="resnet18":
#         model = models.resnet18(pretrained=pretrained)
#     elif name=="resnet34":
#         model = models.resnet34(pretrained=pretrained)
#     elif name=="resnet50":
#         model = models.resnet50(pretrained=pretrained)
#     elif name=="resnet101":
#         model = models.resnet101(pretrained=pretrained)
#     elif name=="resnet152":
#         model = models.resnet152(pretrained=pretrained)

#     return model


def load_model_imagenet(Model, pretrained=False, **kwargs):
    if Model=="Resnet18":
        pretrained_model = models.resnet18(pretrained=pretrained)
    elif Model=="Resnet34":
        pretrained_model = models.resnet34(pretrained=pretrained)
    elif Model=="Resnet50":
        pretrained_model = models.resnet50(pretrained=pretrained)
    elif Model=="Resnet101":
        pretrained_model = models.resnet101(pretrained=pretrained)
    elif Model=="Resnet152":
        pretrained_model = models.resnet152(pretrained=pretrained)
    
    elif Model == 'ConvNeXt_base':
        weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        pretrained_model = models.convnext_base(weights=weights)
    elif Model == 'ConvNeXt_tiny':
        weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        pretrained_model = models.convnext_tiny(weights=weights)
    elif Model == 'ConvNeXt_large':
        weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        pretrained_model = models.convnext_large(weights=weights)
    elif  Model == 'ConvNeXt_small':
        # weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        pretrained_model = models.convnext_small(pretrained=pretrained)
    
    elif Model == 'ViT_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        pretrained_model = models.vit_b_16(pretrained=pretrained)

    elif Model == 'DenseNet121':
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
        pretrained_model = models.densenet121(weights=weights)
    elif Model == 'DenseNet161':
        weights=models.DenseNet161_Weights.IMAGENET1K_V1
        pretrained_model = models.densenet161(weights=weights)
    elif Model == 'DenseNet169':
        weights=models.DenseNet169_Weights.IMAGENET1K_V1
        pretrained_model = models.densenet169(weights=weights)
    elif Model == 'DenseNet201':
        weights=models.DenseNet201_Weights.IMAGENET1K_V1
        pretrained_model = models.densenet201(weights=weights)


    elif Model == 'EfficientNet_B0':
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        pretrained_model = models.efficientnet_b0(weights=weights)


    elif Model == 'Regnet400mf':
        pretrained_model = models.regnet_y_400mf(pretrained=pretrained)
    elif Model == 'Regnet800mf':
        pretrained_model = models.regnet_y_800mf(pretrained=pretrained)
    elif Model == 'Regnet6gf':
        pretrained_model = models.regnet_y_1_6gf(pretrained=pretrained)
    elif Model == 'Regnet2gf':
        pretrained_model = models.regnet_y_3_2gf(pretrained=pretrained)
    elif Model == 'Regnet8gf':
        pretrained_model = models.regnet_y_8gf(pretrained=pretrained)
    elif Model == 'Regnet16gf':
        pretrained_model = models.regnet_y_16gf(pretrained=pretrained)
    elif Model == 'Regnet32gf':
        pretrained_model = models.regnet_y_32gf(pretrained=pretrained)



    elif Model=="Squeezenet":
        pretrained_model = models.squeezenet1_0(pretrained=pretrained)
    elif Model == 'Shufflenetv2':
        pretrained_model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    elif Model == 'Mnasnet':
        pretrained_model = models.mnasnet1_0(pretrained=pretrained)
    elif Model == 'mobilenet_v2':
        pretrained_model = models.mobilenet_v2(pretrained=pretrained)
    elif Model == 'mobilenet_v2_RV2':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
        pretrained_model = models.mobilenet_v2(weights=weights).eval()
    
    elif Model == 'Alexnet':
        pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    elif Model == 'VGG16':
        pretrained_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# --------------------------------- ViT -----------------------------------
    elif Model == 'ViT_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        pretrained_model = models.vit_b_16(pretrained=pretrained)
    elif Model == 'ViT_b_32':
        weights = models.ViT_B_32_Weights.IMAGENET1K_V1
        pretrained_model = models.vit_b_32(pretrained=pretrained)
    elif Model == 'ViT_l_16':
        weights = models.ViT_L_16_Weights.IMAGENET1K_V1
        pretrained_model = models.vit_l_16(pretrained=pretrained)
    elif Model == 'ViT_l_32':
        weights = models.ViT_L_32_Weights.IMAGENET1K_V1
        pretrained_model = models.vit_l_32(pretrained=pretrained)

    elif Model == 'MaxVit_T':
        weights = models.MaxVit_T_Weights.IMAGENET1K_V1
        pretrained_model = models.maxvit_t(pretrained=pretrained)

    elif Model == 'Swin_T':
        weights = models.Swin_T_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_t(pretrained=pretrained)
    elif Model == 'Swin_S':
        weights = models.Swin_S_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_s(pretrained=pretrained)
    elif Model == 'Swin_B':
        weights = models.Swin_B_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_b(pretrained=pretrained)

    elif Model == 'Swin_V2_T':
        weights = models.Swin_V2_T_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_v2_t(pretrained=pretrained)
    elif Model == 'Swin_V2_S':
        weights = models.Swin_V2_S_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_v2_s(pretrained=pretrained)
    elif Model == 'Swin_V2_B':
        weights = models.Swin_V2_B_Weights.IMAGENET1K_V1
        pretrained_model = models.swin_v2_b(pretrained=pretrained)
    
    else: 
        print("Enter a model SOS")
        exit(0)
    pretrained_model = pretrained_model.eval()
    return pretrained_model
