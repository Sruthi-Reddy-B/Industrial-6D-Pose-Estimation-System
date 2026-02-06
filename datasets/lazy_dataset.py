
from torch.utils.data import Dataset
import cv2
import os

class LazyImageDataset(Dataset):
    def __init__(self, image_dir, split_file, transform=None):
        with open(split_file) as f:
            self.ids = [l.strip() for l in f]
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.ids[idx]}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image

