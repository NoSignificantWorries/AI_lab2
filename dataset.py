import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm


class DatasetParam:
    num_classes = 2
    batch_size = 4
    num_workers = 4
    dataset_path = "resources/water_v2"
    images_path = os.path.join(dataset_path, "JPEGImages")
    masks_path = os.path.join(dataset_path, "Annotations")
    train_part = []
    valid_part = []
    mean = [0.4421, 0.4703, 0.4621]
    std = [0.2195, 0.2182, 0.2404]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


class SegmentationDataset(Dataset):
    def __init__(self, image_root_dir, mask_root_dir, part, transform=None):
        self.image_root_dir = image_root_dir
        self.mask_root_dir = mask_root_dir
        self.transform = transform
        self.images = []
        images = []
        for subdir in part:
            images += list(map(lambda x: os.path.join(subdir, x), os.listdir(os.path.join(image_root_dir, subdir))))
        for image_path in images:
            mask_name = image_path.replace(".jpg", ".png")
            mask_path = os.path.join(self.mask_root_dir, mask_name)
            if os.path.exists(mask_path):
                self.images.append(image_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        mask_name = image_name.replace(".jpg", ".png")
        image_path = os.path.join(self.image_root_dir, image_name)
        mask_path = os.path.join(self.mask_root_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path), dtype=np.float64) / 255

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
            mask = transforms.Resize((256, 256))(mask)
            mask = mask.long().squeeze()
            mask = torch.argmax(mask, dim=0)

        return image, mask


def calculate_mean_std(dataset, batch_size, num_workers):
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    with tqdm(dataloader, desc="Calculating progress") as pbar:
        for images in pbar:
            batch_size = images[0].size(0)
            images = images[0].view(batch_size, images[0].size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_size
            
            pbar.set_postfix({"samples": total_samples})

    mean /= total_samples
    std /= total_samples

    return mean, std


def get_mean_std():
    valid = SegmentationDataset(DatasetParam.images_path, DatasetParam.masks_path, DatasetParam.valid_part,
                         transform=transforms.Compose([transforms.Resize([256, 256]),
                                                       transforms.ToTensor()]))
    train = SegmentationDataset(DatasetParam.images_path, DatasetParam.masks_path, DatasetParam.train_part,
                         transform=transforms.Compose([transforms.Resize([256, 256]),
                                                       transforms.ToTensor()]))
    train.images += valid.images
    return calculate_mean_std(dataset=train,
                              batch_size=DatasetParam.batch_size,
                              num_workers=DatasetParam.num_workers)


def get_annotations(file):
    with open(os.path.join(DatasetParam.dataset_path, file), "r") as ann_file:
        results = [line[:-1] for line in ann_file.readlines() if bool(line[:-1])]
    return results
    

def init():
    DatasetParam.train_part = get_annotations("train.txt")
    DatasetParam.valid_part = get_annotations("val.txt")


def main():
    init()
    print(get_mean_std())


if __name__ == "__main__":
    main()
