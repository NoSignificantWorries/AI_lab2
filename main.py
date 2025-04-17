import os
import json
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import numpy as np

import dataset as dset
import model as mdl


def train(device, criterion, optimizer, model, dataloaders, num_epochs, valid_epoch_step, save_period=-1, checkpoint_dir=None):
    save_dir = f"work/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not os.path.exists(f"{save_dir}/weights"):
        os.makedirs(f"{save_dir}/weights")

    if checkpoint_dir is None:
        results = {
            "train": {
                "Loss": [],
                "IoU 0": [],
                "IoU 1": [],
                "avg IoU": [],
                "Dice 0": [],
                "Dice 1": [],
                "avg Dice": []
            },
            "valid": {
                "Loss": [],
                "IoU 0": [],
                "IoU 1": [],
                "avg IoU": [],
                "Dice 0": [],
                "Dice 1": [],
                "avg Dice": []
            }
        }
    else:
        with open(f"work/{checkpoint_dir}/results.json", "r", encoding="utf-8") as file:
            results = json.load(file)

    for epoch in range(num_epochs):
        if (epoch + 1) % valid_epoch_step == 0:
            phase = "valid"
            model.eval()
        else:
            phase = "train"
            model.train()

        running_loss = 0.0
        running_IoU_0 = 0.0
        running_IoU_1 = 0.0
        running_dice_0 = 0.0
        running_dice_1 = 0.0
        total_samples = 0
        with tqdm(dataloaders[phase], desc=f"Epoch: {epoch + 1}/{num_epochs} Phase: {phase}") as pbar:
            for iter_i, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)["out"]
                    
                    loss = criterion(outputs, masks)
                    output_mask = torch.softmax(outputs, dim=1)
                    output_mask = torch.argmax(output_mask, dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                iou = mdl.compute_iou(output_mask, masks)
                dice = mdl.compute_dice(output_mask, masks)
                running_IoU_0 += iou[0] * images.size(0)
                running_IoU_1 += iou[1] * images.size(0)
                running_dice_0 += dice[0] * images.size(0)
                running_dice_1 += dice[1] * images.size(0)

                total_samples += images.size(0)

                avg_IoU = (running_IoU_0 + running_IoU_1) / (2 * total_samples) 
                avg_dice = (running_dice_0 + running_dice_1) / (2 * total_samples) 
                pbar.set_postfix({"loss": f"{(running_loss / total_samples):.3f}",
                                  "IoU(0/1/avg)": f"{(running_IoU_0 / total_samples):.3f}/{(running_IoU_1 / total_samples):.3f}/{(avg_IoU):.3f}",
                                  "dice(0/1/avg)": f"{(running_dice_0 / total_samples):.3f}/{(running_dice_1 / total_samples):.3f}/{(avg_dice):.3f}"})
                
                if (iter_i + 1) % 100 == 0:
                    results[phase]["Loss"].append(running_loss / total_samples)
                    results[phase]["IoU 0"].append(running_IoU_0 / total_samples)
                    results[phase]["IoU 1"].append(running_IoU_1 / total_samples)
                    results[phase]["avg IoU"].append((running_IoU_1 + running_IoU_0) / (2 * total_samples))
                    results[phase]["Dice 0"].append(running_dice_0 / total_samples)
                    results[phase]["Dice 1"].append(running_dice_1 / total_samples)
                    results[phase]["avg Dice"].append((running_dice_0 + running_dice_1) / (2 * total_samples))

                    with open(f"{save_dir}/results.json", "w") as file:
                        json.dump(results, file, indent=4)

        if save_period > 0:
            if (epoch + 1) % save_period == 0:
                torch.save(model.state_dict(), f"{save_dir}/weights/model_{epoch + 1}.pth")
    
    torch.save(model.state_dict(), f"{save_dir}/weights/model_last.pth")

    return results


def test(device, criterion, model, dataloaders):
    model.eval()

    running_loss = 0.0
    running_IoU_0 = 0.0
    running_IoU_1 = 0.0
    running_dice_0 = 0.0
    running_dice_1 = 0.0
    total_samples = 0
    with tqdm(dataloaders["valid"], desc="Testing...") as pbar:
        for (images, masks) in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            output_mask = torch.softmax(outputs, dim=1)
            output_mask = torch.argmax(output_mask, dim=1)

            running_loss += loss.item() * images.size(0)
            iou = mdl.compute_iou(output_mask, masks)
            dice = mdl.compute_dice(output_mask, masks)
            running_IoU_0 += iou[0] * images.size(0)
            running_IoU_1 += iou[1] * images.size(0)
            running_dice_0 += dice[0] * images.size(0)
            running_dice_1 += dice[1] * images.size(0)

            total_samples += images.size(0)

            avg_IoU = (running_IoU_0 + running_IoU_1) / (2 * total_samples) 
            avg_dice = (running_dice_0 + running_dice_1) / (2 * total_samples) 
            pbar.set_postfix({"loss": f"{(running_loss / total_samples):.3f}",
                              "IoU(0/1/avg)": f"{(running_IoU_0 / total_samples):.3f}/{(running_IoU_1 / total_samples):.3f}/{(avg_IoU):.3f}",
                              "dice(0/1/avg)": f"{(running_dice_0 / total_samples):.3f}/{(running_dice_1 / total_samples):.3f}/{(avg_dice):.3f}"})
            

def main(mode="train", num_epochs=10, valid_epoch_step=5, save_period=-1, from_checkpoint=False, chekpoint_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if mode == "test":
        device = "cpu"
    print("Working on device:", device)
    
    train_dataset = dset.SegmentationDataset(dset.DatasetParam.images_path,
                                             dset.DatasetParam.masks_path,
                                             dset.DatasetParam.train_part,
                                             transform=dset.DatasetParam.transform)
    valid_dataset = dset.SegmentationDataset(dset.DatasetParam.images_path,
                                             dset.DatasetParam.masks_path,
                                             dset.DatasetParam.valid_part,
                                             transform=dset.DatasetParam.transform)

    dataloaders = {"train": DataLoader(train_dataset, batch_size=dset.DatasetParam.batch_size, shuffle=True),
                   "valid": DataLoader(valid_dataset, batch_size=dset.DatasetParam.batch_size, shuffle=False)}
    
    model = mdl.init_model(dset.DatasetParam.num_classes, from_checkpoint, f"work/{chekpoint_dir}/weights/model_last.pth").to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=mdl.ModelParam.lr)

    if mode == "train":
        train(device, criterion, optimizer, model, dataloaders, num_epochs, valid_epoch_step, save_period, chekpoint_dir)
    else:
        test(device, criterion, model, dataloaders)


if __name__ == "__main__":
    # main("train", 250, 5, 10, True, "20250415_152102")
    main("test", from_checkpoint=True, chekpoint_dir="20250416_042826")
    '''
    pc = transforms.ToTensor()(np.array([[[1, 0, 2, 0], [1, 0, 0, 3], [1, 0, 0, 0], [1, 0, 0, 0]]]))
    tg = transforms.ToTensor()(np.array([[[1, 1, 2, 0], [1, 1, 2, 0], [1, 1, 0, 0], [1, 1, 0, 0]]]))
    print(mdl.compute_iou(pc, tg))
    print(mdl.compute_dice(pc, tg))
    '''
