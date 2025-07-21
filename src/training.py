import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

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
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256), nn.ReLU(),
            nn.Linear(256, 20) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

# Annahme: 2 Klassen, passe ggf. an!
num_classes = 4

# Dataset laden
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = YoloDataset(
    img_dir="Dataset/fsoco_bounding_boxes_train/amz/img",
    label_dir="Dataset/fsoco_bounding_boxes_train/amz/labels",
    transform=transform
)

def yolo_collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, 0)  # Bilder können gestapelt werden
    return images, targets           # targets bleibt eine Liste von Tensors

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=yolo_collate_fn
)

# Modell
model = YoloNano(num_classes=1)
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
        targets = [t.to(device) for t in targets]
        outputs = model(images)
        # ACHTUNG: Nur für Testzwecke, nicht für echtes Training!
        targets_tensor = torch.zeros(outputs.shape, device=outputs.device)
        for i, t in enumerate(targets):
            # t[0] ist die erste Box, [class, x, y, w, h]
            targets_tensor[i, :5] = t[0]  # Rest bleibt 0
        loss = criterion(outputs, targets_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")