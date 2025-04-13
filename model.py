from torchvision import models
import torch.nn as nn

from dataset import DatasetParam


class ModelParam:
    lr = 0.0001
    batch_size = DatasetParam.batch_size

def init_model(num_classes, pretrained=True, weights="DeepLabV3_ResNet101_Weights.DEFAULT"):
    if pretrained:
        model = models.segmentation.deeplabv3_resnet101(weights=weights, progress=True)
    else:
        model = models.segmentation.deeplabv3_resnet101(progress=True)

    in_features = model.classifier[4].in_channels

    model.classifier[4] = nn.Conv2d(in_features, num_classes, kernel_size=1)
    
    return model
