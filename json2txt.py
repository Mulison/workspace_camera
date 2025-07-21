import os
import json
import glob

root_dir = 'Dataset/fsoco_bounding_boxes_train'

for team in os.listdir(root_dir):
    team_path = os.path.join(root_dir, team)
    ann_path = os.path.join(team_path, 'ann')
    if os.path.isdir(ann_path):
        for json_file in glob.glob(os.path.join(ann_path, '*.json')):
            # Hier deine bisherige Verarbeitung für jede JSON-Datei
            print(f"Verarbeite: {json_file}")
            json_path = json_file.replace('\\', '/')
            txt_filename = json_file.replace('.png.json', '.txt').replace('.jpg.json', '.txt')
            txt_path = os.path.join(ann_path, txt_filename)
            txt_path = txt_path.replace('\\', '/')

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Passe diese Zeilen an dein JSON-Format an!
            # ACHTUNG: img_w und img_h waren vertauscht!
            img_w = data['size']['width']
            img_h = data['size']['height']
            objects = data['objects']

            with open(txt_path, 'w') as out:
                for obj in objects:
                    # Hier sollte ein Mapping von classTitle zu class_id erfolgen!
                    class_id = obj['classId']  # Platzhalter, ggf. Mapping nutzen
                    x_min, y_min = obj['points']['exterior'][0]
                    x_max, y_max = obj['points']['exterior'][1]
                    x_center = ((x_min + x_max) / 2) / img_w
                    y_center = ((y_min + y_max) / 2) / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h
                    out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
