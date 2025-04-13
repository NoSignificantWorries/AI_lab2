import os

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
    
    output_mask *= 255
    
    resized_mask = cv2.resize(output_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"results/{image_name}", resized_mask)


def main(weights):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = mdl.init_model(dset.DatasetParam.num_classes, True, weights).to(device)
    model.eval()
    
    with tqdm(os.listdir("resources/water_v2/JPEGImages/ADE20K"), desc="Working...") as pbar:
        for image_name in pbar:
            run_image(device, model, f"resources/water_v2/JPEGImages/ADE20K/{image_name}", image_name)


if __name__ == "__main__":
    main("work/20250413_124313/weights/model_last.pth")
