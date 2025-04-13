import os
import json
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
# import kagglehub

import dataset as dset
dset.init()
import model as mdl


def train(device, criterion, optimizer, model, dataloaders, num_epochs, valid_epoch_step, save_period=-1):
    save_dir = f"work/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not os.path.exists(f"{save_dir}/weights"):
        os.makedirs(f"{save_dir}/weights")

    results = {
        "train": {
            "Loss": [],
            "IoU": [],
            "Dice": []
        },
        "valid": {
            "Loss": [],
            "IoU": [],
            "Dice": []
        }
    }

    for epoch in range(num_epochs):
        if (epoch + 1) % valid_epoch_step == 0:
            phase = "valid"
            model.eval()
        else:
            phase = "train"
            model.train()

        running_loss = 0.0
        running_IoU = 0.0
        running_dice = 0.0
        total_samples = 0
        with tqdm(dataloaders[phase], desc=f"Epoch: {epoch + 1}/{num_epochs} Phase: {phase}") as pbar:
            for (images, masks) in pbar:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                        outputs = model(images)["out"]
                        loss = criterion(outputs, masks)
                        output_mask = torch.softmax(outputs, dim=1)
                        output_mask = torch.argmax(output_mask, dim=1)

                        # updating weights if phase value is "train"
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_IoU += mdl.calculate_IoU(output_mask, masks) * images.size(0)
                running_dice += mdl.calculate_dice(output_mask, masks) * images.size(0)

                total_samples += images.size(0)
                    
                pbar.set_postfix({"loss": running_loss / total_samples,
                                  "IoU": running_IoU / total_samples,
                                  "dice": running_dice / total_samples})
                
        print(f"Total metrics:\nLoss: {running_loss / total_samples}, IoU: {running_IoU / total_samples}, Dice: {running_dice / total_samples}")

        results[phase]["Loss"].append(running_loss / total_samples)
        results[phase]["IoU"].append(running_IoU / total_samples)
        results[phase]["Dice"].append(running_dice / total_samples)

        if save_period > 0:
            if (epoch + 1) % save_period == 0:
                torch.save(model.state_dict(), f"{save_dir}/weights/model_{epoch + 1}.pth")
    
    torch.save(model.state_dict(), f"{save_dir}/weights/model_last.pth")

    # saving training data in json file
    with open(f"{save_dir}/results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)
    return results


def test(device, criterion, model, dataloaders):
    model.eval()

    running_loss = 0.0
    running_IoU = 0.0
    running_dice = 0.0
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
            running_IoU += mdl.calculate_IoU(output_mask, masks) * images.size(0)
            running_dice += mdl.calculate_dice(output_mask, masks) * images.size(0)

            total_samples += images.size(0)

            pbar.set_postfix({"loss": running_loss / total_samples,
                              "IoU": running_IoU / total_samples,
                              "dice": running_dice / total_samples})
            
    print(f"Total metrics:\nLoss: {running_loss / total_samples}, IoU: {running_IoU / total_samples}, Dice: {running_dice / total_samples}")



def main(mode="train", num_epochs=10, valid_epoch_step=5, save_period=-1, weights="work"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
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
                   "valid": DataLoader(valid_dataset, batch_size=dset.DatasetParam.batch_size, shuffle=True)}
    
    if mode == "train":
        model = mdl.init_model(dset.DatasetParam.num_classes).to(device)
    else:
        model = mdl.init_model(dset.DatasetParam.num_classes, True, weights).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=mdl.ModelParam.lr)

    if mode == "train":
        train(device, criterion, optimizer, model, dataloaders, num_epochs, valid_epoch_step, save_period)
    else:
        test(device, criterion, model, dataloaders)


if __name__ == "__main__":
    main("train", 30, 5, 5)
    # main("test", weights="work/20250413_124313/weights/model_last.pth")
