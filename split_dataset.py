import os
import shutil
import random

# Parameter
SOURCE_DIR = 'Dataset'
TARGET_DIR = 'Dataset_split'
TRAIN_RATIO = 0.8
IMG_SUBDIR = 'img'
LABELS_SUBDIR = 'labels'

# Zielordner anlegen
def make_dirs():
    for split in ['train', 'test']:
        os.makedirs(os.path.join(TARGET_DIR, split, IMG_SUBDIR), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, split, LABELS_SUBDIR), exist_ok=True)

def collect_all_files():
    pairs = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        img_dir = os.path.join(root, IMG_SUBDIR)
        img_dir = img_dir.replace('\\', '/')
        labels_dir = os.path.join(root, LABELS_SUBDIR)
        labels_dir = labels_dir.replace('\\', '/')
        if os.path.isdir(img_dir) and os.path.isdir(labels_dir):
            for img_file in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_file)
                img_path = img_path.replace('\\', '/')
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                label_path = label_path.replace('\\', '/')
                if os.path.isfile(img_path) and os.path.isfile(label_path):
                    pairs.append((img_path, label_path))
    return pairs

def split_and_copy(pairs):
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    for split, split_pairs in [('train', train_pairs), ('test', test_pairs)]:
        for img_path, label_path in split_pairs:
            img_filename = os.path.basename(img_path)
            label_filename = os.path.basename(label_path)
            shutil.copy2(img_path, os.path.join(TARGET_DIR, split, IMG_SUBDIR, img_filename))
            shutil.copy2(label_path, os.path.join(TARGET_DIR, split, LABELS_SUBDIR, label_filename))

def main():
    make_dirs()
    pairs = collect_all_files()
    print(f'Gefundene Bild-Label-Paare: {len(pairs)}')
    split_and_copy(pairs)
    print('Fertig!')

if __name__ == '__main__':
    main() 