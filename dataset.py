import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import kagglehub


class DatasetParam:
    num_classes = 2
    batch_size = 2
    num_workers = 4
    img_size = 512
    dataset_path = "resources/water_v2"
    images_path = os.path.join(dataset_path, "JPEGImages")
    masks_path = os.path.join(dataset_path, "Annotations")
    train_part = ["ADE20K", "river_segs"]
    valid_part = [
        "aberlour",
        "auldgirth",
        "bewdley",
        "cockermouth",
        "dublin",
        "evesham-lock",
        "galway-city",
        "holmrook",
        "keswick_greta",
        "worcester",
        "stream0",
        "stream1",
        "stream3_small",
        "stream2",
        "boston_harbor2_small_rois",
        "buffalo0_small",
        "canal0",
        "mexico_beach_clip0",
        "holiday_inn_clip0",
        "gulf_crest",
    ]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


class SegmentationDataset(Dataset):
    def __init__(self, image_root_dir, mask_root_dir, subdirs, transform=None):
        self.image_root_dir = image_root_dir
        self.mask_root_dir = mask_root_dir
        self.transform = transform

        self.images = []
        for subdir in subdirs:
            for image_name in os.listdir(f"{image_root_dir}/{subdir}"):
                if os.path.exists(f"{mask_root_dir}/{subdir}/{image_name.replace('.jpg', '.png')}"):
                    if os.path.exists(f"{image_root_dir}/{subdir}/{image_name}"):
                        self.images.append(f"{subdir}/{image_name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        mask_name = image_name.replace(".jpg", ".png")
        image_path = os.path.join(self.image_root_dir, image_name)
        mask_path = os.path.join(self.mask_root_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path), dtype=np.float64)
        mask = np.where(mask > 0, 1, 0)[:, :, 0]

        image = self.transform(image)
        mask = transforms.ToTensor()(mask)
        mask = transforms.Resize((DatasetParam.img_size, DatasetParam.img_size))(mask)
        mask = mask.long().squeeze()
        
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
    data = SegmentationDataset(DatasetParam.images_path, DatasetParam.masks_path,
                         transform=transforms.Compose([transforms.Resize([DatasetParam.img_size, DatasetParam.img_size]),
                                                       transforms.ToTensor()]))
    return calculate_mean_std(dataset=data,
                              batch_size=DatasetParam.batch_size,
                              num_workers=DatasetParam.num_workers)


def main():
    print(get_mean_std())


if __name__ == "__main__":
    main()
