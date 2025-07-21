import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

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

# Annahme: 2 Klassen, passe ggf. an!
num_classes = 2

# Dataset laden
train_dataset = YoloDataset(
    img_dir="Dataset/fsoco_bounding_boxes_train/amz/img",
    label_dir="Dataset/fsoco_bounding_boxes_train/amz/labels"
    # img_dir="Dataset_split/test/img",
    # label_dir="Dataset_split/test/labels"
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Modell
model = YoloNano(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss und Optimizer (Dummy-Loss, für echtes Training YOLO-Loss verwenden!)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(10):
    model.train()
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets.view(outputs.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")