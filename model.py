from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import DatasetParam

# 1094
# 106

class ModelParam:
    lr = 0.001
    batch_size = DatasetParam.batch_size

def init_model(num_classes, pretrained=False, weights_path="work"):
    model = models.segmentation.deeplabv3_resnet50(weights="DeepLabV3_ResNet50_Weights.DEFAULT", progress=True)
    
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    if pretrained:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint)
    
    return model

def compute_iou(pred, target):
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=DatasetParam.num_classes).permute(0, 3, 1, 2)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=DatasetParam.num_classes).permute(0, 3, 1, 2)

    intersection = (pred_one_hot & target_one_hot).sum(dim=(0, 2, 3)).float()
    union = (pred_one_hot | target_one_hot).sum(dim=(0, 2, 3)).float()

    iou = intersection / (union + 1e-6)
    
    return iou[0].item(), iou[1].item()

def compute_dice(pred, target):
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=DatasetParam.num_classes).permute(0, 3, 1, 2)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=DatasetParam.num_classes).permute(0, 3, 1, 2)

    intersection = (pred_one_hot & target_one_hot).sum(dim=(0, 2, 3)).float()

    dice = (2 * intersection) / (pred_one_hot.sum(dim=(0, 2, 3)).float() + target_one_hot.sum(dim=(0, 2, 3)).float() + 1e-6)

    return dice[0].item(), dice[1].item()
