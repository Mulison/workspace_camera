import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        image = Image.open(img_path).convert("RGB")
    boxes = []
        with open(label_path) as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                boxes.append([class_id, x_center, y_center, width, height])
        boxes = torch.tensor(boxes)
        if self.transform:
            image = self.transform(image)
        return image, boxes

class YoloNano(nn.Module):
    def __init__(self, num_classes):
        super(YoloNano, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256), nn.ReLU(),
            nn.Linear(256, num_classes * 5)  # 5 = [class, x, y, w, h]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x