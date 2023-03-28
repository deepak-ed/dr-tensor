"""
@description: Dataset class for femur dataset. It can be used with pytorch Dataloaders.
Imgaug is used for online augmentations.
https://github.com/aleju/imgaug
It store only one image in memory because it load image and label each time when it is accessed.
Images are resized and normalized as well.

@author: Muhammad Abdullah
"""
import csv
import json

import imgaug.augmenters as ia
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch

from Backend.network.utils.math_funcs import rect_line


class Dataset(object):
    def __init__(self, path) -> None:
        self.path = path
        self.data = []
        self.initialize_data(path)
        self.transform = True
        self.should_normalize = True
        self.img_aug =  ia.Sometimes(0.8, ia.SomeOf(2, [
            ia.LinearContrast((0.75, 1.5)),
            ia.Affine(scale=(0.7, 1.5)),
            ia.Affine(translate_px={"x": (-200, 200), "y": (-200, 200)}),
            ia.Affine(rotate=(-45, 45)),
        ]))

    def __getitem__(self, i) -> dict:
        item = self.data[i]
        self.item = item
        img, resize_factor = self._load_img(item["image"], resize=True, new_size=(512, 512))
        self._load_labels(item["label"], resize_factor)
        
        #shaft points
        shaft_rl = self._load_coords("Femoral Shaft Lateral Right")
        shaft_rm = self._load_coords("Femoral Shaft Medial Right")
        shaft_rc = self._load_coords("Femoral Shaft Centerline Right")
        shaft_ll = self._load_coords("Femoral Shaft Lateral Left")
        shaft_lm = self._load_coords("Femoral Shaft Medial Left")
        shaft_lc = self._load_coords("Femoral Shaft Centerline Left")

        # neck points
        neck_rl = self._load_coords("Femoral Neck Lateral Right")
        neck_rm = self._load_coords("Femoral Neck Medial Right")
        neck_rc = self._load_coords("Femoral Neck Centerline Right")
        neck_ll = self._load_coords("Femoral Neck Lateral Left")
        neck_lm = self._load_coords("Femoral Neck Medial Left")
        neck_lc = self._load_coords("Femoral Neck Centerline Left")

        
        points = np.concatenate((shaft_rl,
                                    shaft_rm,
                                    shaft_rc,
                                    shaft_ll,
                                    shaft_lm,
                                    shaft_lc,
                                    neck_rl,
                                    neck_rm,
                                    neck_rc,
                                    neck_ll,
                                    neck_lm,
                                    neck_lc), axis=0)

        kps = np.array([points])
        
        if self.transform:
            img, kps = self.perform_augmentation(img, kps)

        points = kps[0]
        shaft_rl = points[0:2]
        shaft_rm = points[2:4]
        shaft_rc = points[4:6]
        shaft_ll = points[6:8]
        shaft_lm = points[8:10]
        shaft_lc = points[10:12]
        neck_rl = points[12:14]
        neck_rm = points[14:16]
        neck_rc = points[16:18]
        neck_ll = points[18:20]
        neck_lm = points[20:22]
        neck_lc = points[22:]

        shaft_rl_map = rect_line(img.shape, np.flip(shaft_rl[0], 0), np.flip(shaft_rl[1], 0), sigma=5, text="shaft_rl+"+item["label"])
        shaft_rm_map = rect_line(img.shape, np.flip(shaft_rm[0], 0), np.flip(shaft_rm[1], 0), sigma=5, text="shaft_rm+"+item["label"])
        shaft_rc_map = rect_line(img.shape, np.flip(shaft_rc[0], 0), np.flip(shaft_rc[1], 0), sigma=5, text="shaft_rc+"+item["label"])
        shaft_ll_map = rect_line(img.shape, np.flip(shaft_ll[0], 0), np.flip(shaft_ll[1], 0), sigma=5, text="shaft_ll+"+item["label"])
        shaft_lm_map = rect_line(img.shape, np.flip(shaft_lm[0], 0), np.flip(shaft_lm[1], 0), sigma=5, text="shaft_lm+"+item["label"])
        shaft_lc_map = rect_line(img.shape, np.flip(shaft_lc[0], 0), np.flip(shaft_lc[1], 0), sigma=5, text="shaft_lc+"+item["label"])

        neck_rl_map = rect_line(img.shape, np.flip(neck_rl[0], 0), np.flip(neck_rl[1], 0), sigma=5, text="neck_rl+"+item["label"])
        neck_rm_map = rect_line(img.shape, np.flip(neck_rm[0], 0), np.flip(neck_rm[1], 0), sigma=5, text="neck_rm+"+item["label"])
        neck_rc_map = rect_line(img.shape, np.flip(neck_rc[0], 0), np.flip(neck_rc[1], 0), sigma=5, text="neck_rc+"+item["label"])
        neck_ll_map = rect_line(img.shape, np.flip(neck_ll[0], 0), np.flip(neck_ll[1], 0), sigma=5, text="neck_ll+"+item["label"])
        neck_lm_map = rect_line(img.shape, np.flip(neck_lm[0], 0), np.flip(neck_lm[1], 0), sigma=5, text="neck_lm+"+item["label"])
        neck_lc_map = rect_line(img.shape, np.flip(neck_lc[0], 0), np.flip(neck_lc[1], 0), sigma=5, text="neck_lc+"+item["label"])


        femur = np.array([shaft_rl_map,
                            shaft_rm_map,
                            shaft_rc_map,
                            shaft_ll_map,
                            shaft_lm_map,
                            shaft_lc_map,
                            neck_rl_map,
                            neck_rm_map,
                            neck_rc_map,
                            neck_ll_map,
                            neck_lm_map,
                            neck_lc_map])

        if self.should_normalize:
            img = self.normalize(img)
        # img to tensor
        img = np.reshape(img, (1, *img.shape))
        img = img.astype(dtype=np.float32)
        img = torch.from_numpy(img)

        # points to tensor
        points = points.astype(dtype=np.float32)
        points = torch.from_numpy(points)

        # femur to tensor
        femur = femur.astype(dtype=np.float32)
        femur = torch.from_numpy(femur)

        item = {"image": img, "femur_map": femur, "points": points, "patient_id": item["patient_id"]}
        return item

    def __len__(self):
        return len(self.data)
    
    def _load_img(self, path, resize=True, new_size=None) -> Image:
        if resize:
            if new_size is None or not isinstance(new_size, tuple):
                raise ValueError("Specify new_size of tuple (new_width, new_height)")

        img = Image.open(path)
        img = img.convert(mode="L")
        if not resize:
            return img, np.ones((1, 2))
        w, h = img.width, img.height
        new_w, new_h = new_size
        background = Image.new("L", (new_w, new_h))
        img.thumbnail((new_w, new_h))
        dw, dh = img.size
        df = np.array([dw/w, dh/h])
        background.paste(img, (0,0))
        img = background
        img = np.asarray(img, dtype=np.float32)
        return img, df

    def _load_labels(self, path, resize_factor):
        with open(path) as fp:
            self.orig_labels = json.load(fp)
            for shape in self.orig_labels["shapes"]:
                points = np.array(shape["points"], dtype=np.float32)
                points *= resize_factor
                shape["points"] = points
            

    def _load_coords(self, type_) -> np.array:
            data = self.orig_labels
            for shape in data["shapes"]:
                if shape["label"] == type_:
                    points = np.array(shape["points"], dtype=np.float32)
                    return points
            raise ValueError(f"{type_} is not found in labels at {self.item['label']}")

    def initialize_data(self, path) -> None:
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                item = dict()
                item["image"] = row[0]
                item["label"] = row[1]
                item["patient_id"] = row[2]
                item["id"] = len(self.data)
                self.data.append(item)

    def perform_augmentation(self, img, kps):
        img, kps = self.img_aug(image=img, keypoints=kps)
        return img, kps

    def normalize(self, item):
        mean = np.mean(item)
        std = np.std(item)
        norm_item = (item - mean)/std
        return norm_item

if __name__=="__main__":
    csv_path = "train.csv"
    dataset = Dataset(path=csv_path)
    dataset.should_normalize = False
    import time
    for i in range(len(dataset)):
        item = dataset[i]
        femur = item["femur_map"].numpy()
        image = item["image"].numpy()[0]
        plt.imshow(image, cmap="gray")    
        plt.imshow(np.sum(femur, 0), alpha=0.5, cmap="gray")
        print(i, dataset.item["label"])
        plt.savefig("visualisations/femur.png")
        time.sleep(5)
