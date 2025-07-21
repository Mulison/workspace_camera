import os
import shutil
import random

# Verzeichnisse anpassen
base_dir = os.path.join(os.path.dirname(__file__), 'Dataset/fsoco_bounding_boxes_train/amz').replace('\\', '/')
img_dir = os.path.join(base_dir, 'img').replace('\\', '/')
labels_dir = os.path.join(base_dir, 'labels').replace('\\', '/')

img_train_dir = os.path.join(base_dir, 'img_train').replace('\\', '/')
img_test_dir = os.path.join(base_dir, 'img_test').replace('\\', '/')
labels_train_dir = os.path.join(base_dir, 'labels_train').replace('\\', '/')
labels_test_dir = os.path.join(base_dir, 'labels_test').replace('\\', '/')

# Zielordner anlegen
for d in [img_train_dir, img_test_dir, labels_train_dir, labels_test_dir]:
    os.makedirs(d, exist_ok=True)

# Alle Bilddateien auflisten
img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f).replace('\\', '/'))]

# Paare aus Bild und Label bilden (nur wenn beide existieren)
pairs = []
for img_file in img_files:
    name, _ = os.path.splitext(img_file)
    label_file = name + '.txt'
    if os.path.exists(os.path.join(labels_dir, label_file).replace('\\', '/')):
        pairs.append((img_file, label_file))

# Mischen und splitten
random.shuffle(pairs)
split_idx = int(0.8 * len(pairs))
train_pairs = pairs[:split_idx]
test_pairs = pairs[split_idx:]

# Kopieren
for img_file, label_file in train_pairs:
    shutil.copy(os.path.join(img_dir, img_file).replace('\\', '/'), os.path.join(img_train_dir, img_file).replace('\\', '/'))
    shutil.copy(os.path.join(labels_dir, label_file).replace('\\', '/'), os.path.join(labels_train_dir, label_file).replace('\\', '/'))

for img_file, label_file in test_pairs:
    shutil.copy(os.path.join(img_dir, img_file).replace('\\', '/'), os.path.join(img_test_dir, img_file).replace('\\', '/'))
    shutil.copy(os.path.join(labels_dir, label_file).replace('\\', '/'), os.path.join(labels_test_dir, label_file).replace('\\', '/'))

print(f"Fertig! {len(train_pairs)} Trainings- und {len(test_pairs)} Testpaare kopiert.") 