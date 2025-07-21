import os
import cv2
import matplotlib.pyplot as plt

# Beispiel: Team und Dateiname
team = 'amz'
img_dir = f'Dataset/fsoco_bounding_boxes_train/{team}/img'
label_dir = f'Dataset/fsoco_bounding_boxes_train/{team}/labels'

# Beispiel-Datei
img_filename = 'amz_00100.jpg'
txt_filename = img_filename.replace('.png', '.txt').replace('.jpg', '.txt')

img_path = os.path.join(img_dir, img_filename)
img_path = img_path.replace('\\', '/')
txt_path = os.path.join(label_dir, txt_filename)
txt_path = txt_path.replace('\\', '/')

# Bild laden
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# Labels laden und zeichnen
with open(txt_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, width, height = map(float, parts)
        # Umrechnen auf Pixel
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        # Rechteck zeichnen
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        # Optional: class_id anzeigen
        cv2.putText(image, str(int(class_id)), (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

# Bild anzeigen
plt.imshow(image)
plt.axis('off')
plt.show()