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


def calculate_IoU(pred_mask, target_mask, smooth=1e-6):
    intersection = (pred_mask * target_mask).sum(dim=(1, 2))
    union = (pred_mask + target_mask).sum(dim=(1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def calculate_dice(pred_mask, target_mask, smooth=1e-6):
    intersection = (pred_mask * target_mask).sum(dim=(1, 2))
    dice = (2 * intersection + smooth) / ((pred_mask + target_mask).sum(dim=(1, 2)) + smooth)
    return dice.mean()


def calculate_metrics(pred_mask, target_mask, smooth=1e-6):
    TP = ((pred_mask == 1) & (target_mask == 1)).sum(dim=(1, 2)).float()
    FP = ((pred_mask == 1) & (target_mask == 0)).sum(dim=(1, 2)).float()
    FN = ((pred_mask == 0) & (target_mask == 1)).sum(dim=(1, 2)).float()

    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    
    return precision, recall
