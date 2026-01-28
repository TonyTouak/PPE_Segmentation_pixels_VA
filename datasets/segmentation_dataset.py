import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(512, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

        self.img_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])

        self.mask_transform = T.Compose([
            T.Resize(img_size, interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        mask = np.array(mask)
        mask = (mask > 127).astype(np.int64)  # 0 = obstrué, 1 = traversable
        mask = torch.from_numpy(mask)

        return image, mask
