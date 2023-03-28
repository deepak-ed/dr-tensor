"""
@description: This is helper file which create csv file (train_img_labels_paths.csv) from folder containing images and labels.
It assumes that images and corresponding labels are placed side by side with same name.
It assumes images as PNG files and labels as JSON files
@TODO 
change it to make it independent from image type i-e image can be jpeg, png, bmp etc

@author: Muhammad Abdullah
"""
import csv
import glob
import os
import re
import shutil

import numpy as np

ROOT_DIR = "/proj/ciptmp/ic33axaq/IIML/data/"
dataset_path = "projiiml-wise22/data"
DATASET_PATH = os.path.join(ROOT_DIR, dataset_path)

pattern = DATASET_PATH + "/**/*.json"
labels = glob.glob(pathname=pattern, recursive=True)

uniq_ids = {}
for label in labels:
    patient_id = re.search("PA[0-9]{6}", label).group()
    folder_name = re.search("Becken[0-9]{4} [0-9]{1}|Becken[0-9]{4}|Offene Bilder [0-9]{1}|Offene Bilder", label).group()
    unique_id = (folder_name + "-" + patient_id).replace(" ", "_")
    img_path = label.replace(".json", ".png")
    if os.path.exists(img_path):
        if unique_id in uniq_ids.keys():
            uniq_ids[unique_id].append([img_path, label])
        else:
            uniq_ids[unique_id] = []
            uniq_ids[unique_id].append([img_path, label])
    else:
        print(f"Image don't exist for label {label}")

keys = list(uniq_ids.keys())
np.random.shuffle(keys)

folds = np.array_split(keys, 10)
test_fold = folds[0]
val_fold = folds[1]
train_fold = np.concatenate(folds[2:])
print("Patients in test ", len(test_fold))
print("Patients in val ", len(val_fold))
print("Patients in train ", len(train_fold))

with open('train.csv', 'w', newline='') as labelcsv:
    labelwriter = csv.writer(labelcsv, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for k in train_fold:
        for v in uniq_ids[k]:
            labelwriter.writerow([*v, k])

with open('test.csv', 'w', newline='') as labelcsv:
    labelwriter = csv.writer(labelcsv, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for k in test_fold:
        for i, v in enumerate(uniq_ids[k]):
            shutil.copy(v[0], f"test_dataset/{k}_{i}.png")
            labelwriter.writerow([*v, k])

with open('val.csv', 'w', newline='') as labelcsv:
    labelwriter = csv.writer(labelcsv, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for k in val_fold:
        for i, v in enumerate(uniq_ids[k]):
            labelwriter.writerow([*v, k])
