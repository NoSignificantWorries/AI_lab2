import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
# import kagglehub

import dataset as dset
dset.init()
import model as mdl


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Working on device:", device)
    
    dataset = dset.SegmentationDataset(dset.DatasetParam.images_path,
                                       dset.DatasetParam.masks_path,
                                       dset.DatasetParam.train_part,
                                       transform=dset.DatasetParam.transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = mdl.init_model(dset.DatasetParam.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=mdl.ModelParam.lr)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % num_epochs == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/num_epochs:.4f}")
                running_loss = 0.0


if __name__ == "__main__":
    main()
