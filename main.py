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
    for epoch in range(num_epochs):
        if epoch % valid_epoch_step == 0:
            phase = "valid"
            model.eval()
        else:
            phase = "train"
            model.train()

        running_loss = 0.0
        running_IoU = 0.0
        running_dice = 0.0
        running_precision = 0.0
        running_recall = 0.0
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
                prec, rec = mdl.calculate_metrics(output_mask, masks)
                running_precision += prec * images.size(0)
                running_recall += rec * images.size(0)

                total_samples += images.size(0)
                
                pbar.set_postfix({"loss": running_loss / total_samples,
                                  "IoU": running_IoU / total_samples,
                                  "dice": running_dice / total_samples,
                                  "prec": running_precision / total_samples,
                                  "rec": running_recall / total_samples})


def test():
    pass


def main(num_epochs, valid_epoch_step, save_period=-1, mode="train"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    model = mdl.init_model(dset.DatasetParam.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=mdl.ModelParam.lr)

    if mode == "train":
        train(device, criterion, optimizer, model, dataloaders, num_epochs, valid_epoch_step, save_period)
    else:
        test()


if __name__ == "__main__":
    main(30, 10, 5)
