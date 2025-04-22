import os
import random

from tqdm import tqdm
import torch
from PIL import Image
import cv2
from torchvision import transforms
import numpy as np

import dataset as dset
import model as mdl


def run_image(device, model, image_path, image_name):
    orig_image = Image.open(image_path).convert("RGB")
    image = transforms.ToTensor()(orig_image)
    height, width = image.size(1), image.size(2)
    image = dset.DatasetParam.transform(orig_image).to(device)
    image = image.unsqueeze(0)

    outputs = model(image)["out"]
    output_mask = torch.softmax(outputs, dim=1)
    output_mask = torch.argmax(output_mask, dim=1)
    output_mask = output_mask.cpu().numpy()
    output_mask = output_mask[0]
    output_mask *= 255

    resized_mask = cv2.resize(output_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    color = [255, 255, 0]
    color_mask = np.zeros_like(np.array(orig_image), dtype=np.uint8)
    color_mask[resized_mask > 0] = color

    orig_cv = np.array(orig_image)
    orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_RGB2BGR)

    alpha = 0.6
    overlayed = cv2.addWeighted(orig_cv, 1 - alpha, color_mask, alpha, 0)
    
    cv2.imwrite(f"results/{image_name}", overlayed)


def main(weights):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = mdl.init_model(dset.DatasetParam.num_classes, True, weights).to(device)
    model.eval()
    
    images = []
    for dir_ in dset.DatasetParam.valid_part:
        for image_name in os.listdir(os.path.join("resources", "water_v2", "JPEGImages", dir_)):
            images.append((os.path.join("resources", "water_v2", "JPEGImages", dir_, image_name), image_name))
    random.shuffle(images)
    images = images[:int(len(images) * 0.5)]
    
    with tqdm(images, desc="Working...") as pbar:
        for image_path, image_name in pbar:
            run_image(device, model, image_path, image_name)


if __name__ == "__main__":
    main("model/model_last.pth")
