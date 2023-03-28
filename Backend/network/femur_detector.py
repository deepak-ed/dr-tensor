import copy

from matplotlib import pyplot
import numpy as np
from PIL import Image
from sklearn import linear_model
import torch

from Backend.network.network import UNetProd
from Backend.network.unet import UNet


class FemurDetector(object):
    def __init__(self, model, img_size, channels_map):
        self.model = model
        self.net_img_size = img_size
        self.channels_map = channels_map
        self.min_samples_ransac = 10
        self.sigmoid_cutoff = 0.95
        self.batch_size = 1

    def load_model(self, model_path, device="cpu"):
        print("Loading femur detector ...")
        print(f"Using {device} for inference")
        self.model = UNetProd(self.model)
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def process(self, inp):
        inp, df = self._preprocessing(inp)
        out = self._inference(inp)
        out = out[0]
        channels_map = self._postprocessing(out)
        channels_map = self._rescale(channels_map, df)
        output = self._convert_output_format(channels_map)
        return output

    def _preprocessing(self, inp):
        img, df = self._load_and_resize_image(inp)
        img = self._normalize(img)
        img = np.reshape(img, (1, 1, *img.shape))  # NCHW
        img = img.astype(dtype=np.float32)
        return img, df

    def _inference(self, inp):
        print("Received request for inference ...")
        inp = torch.from_numpy(inp)
        y = self.model(inp)
        y = y.detach().numpy()
        return y

    def _postprocessing(self, inp):
        # cutoff based on sigmoid
        inp = 1 / (1 + np.exp(-inp))
        inp[inp < self.sigmoid_cutoff] = 0
        channels_map = copy.deepcopy(self.channels_map)
        channels = inp.shape[0]
        # RANSAC
        for i in range(channels):
            single_channel_pred_mask = inp[i]
            ransac_x, ransac_y = np.where(single_channel_pred_mask > 0)
            # Min length is 2. But 10 is restrictive criterion
            if len(ransac_x) < self.min_samples_ransac:
                channels_map[i]["points"] = None
                continue
            # Robustly fit linear model with RANSAC algorithm
            ransac = linear_model.RANSACRegressor()
            ransac.fit(ransac_x.reshape(-1, 1), ransac_y.reshape(-1, 1))
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            # Remove outliers using Ransac
            outlier_x = ransac_x[outlier_mask]
            outlier_y = ransac_y[outlier_mask]
            indices = np.column_stack((outlier_x, outlier_y))
            for index in indices:
                single_channel_pred_mask[index[0]][index[1]] = 0
            # two point calculation using ransac
            inlier_x = ransac_x[inlier_mask]
            inlier_y = ransac_y[inlier_mask]
            y1 = np.argmax(inlier_x)
            y2 = np.argmin(inlier_x)
            x1 = inlier_x[y1]
            x2 = inlier_x[y2]

            [[y1], [y2]] = ransac.predict(np.array([[x1], [x2]]))
            channels_map[i]["points"] = [[x1, y1], [x2, y2]]

        return channels_map
    
    def _convert_output_format(self, output):
        right_femur_lines = ["Femoral Shaft Lateral Right",
                            "Femoral Shaft Medial Right",
                            "Femoral Shaft Centerline Right",
                            "Femoral Neck Lateral Right",
                            "Femoral Neck Medial Right",
                            "Femoral Neck Centerline Right"]

        left_femur_lines = ["Femoral Shaft Lateral Left",
                                "Femoral Shaft Medial Left",
                                "Femoral Shaft Centerline Left",
                                "Femoral Neck Lateral Left",
                                "Femoral Neck Medial Left",
                                "Femoral Neck Centerline Left"]
        post_output = {}
        post_output["Right"] = {}
        post_output["Left"] = {}

        channels = output.keys()
        for channel in channels:
            if output[channel]["points"] is not None:
                output[channel]["points"] = output[channel]["points"].reshape(-1)
            if output[channel]["name"] in right_femur_lines:
                post_output["Right"][output[channel]["name"]] = output[channel]["points"]
            if output[channel]["name"] in left_femur_lines:
                post_output["Left"][output[channel]["name"]] = output[channel]["points"]
        
        return post_output

    def _load_and_resize_image(self, img: Image):
        img = img.convert(mode="L")
        w, h = img.width, img.height
        new_w, new_h = self.net_img_size
        background = Image.new("L", (new_w, new_h))
        img.thumbnail((new_w, new_h))
        dw, dh = img.size
        df = np.array([dw / w, dh / h])
        background.paste(img, (0, 0))
        img = background
        img = np.asarray(img, dtype=np.float32)
        return img, df

    def _normalize(self, item: np.ndarray):
        mean = np.mean(item)
        std = np.std(item)
        norm_item = (item - mean) / std
        return norm_item

    def _rescale(self, channels_map, df):
        channels = channels_map.keys()
        for channel in channels:
            if channels_map[channel]["points"] is not None:
                channels_map[channel]["points"] /= df
                channels_map[channel]["points"] = np.rint(channels_map[channel]["points"])
        return channels_map


def load_model(model_path):
    unet = UNet(num_classes=12, in_channels=1)
    img_size = (512, 512)
    channels_map = {
        0: {"name": "Femoral Shaft Lateral Right"},
        1: {"name": "Femoral Shaft Medial Right"},
        2: {"name": "Femoral Shaft Centerline Right"},
        3: {"name": "Femoral Shaft Lateral Left"},
        4: {"name": "Femoral Shaft Medial Left"},
        5: {"name": "Femoral Shaft Centerline Left"},
        6: {"name": "Femoral Neck Lateral Right"},
        7: {"name": "Femoral Neck Medial Right"},
        8: {"name": "Femoral Neck Centerline Right"},
        9: {"name": "Femoral Neck Lateral Left"},
        10: {"name": "Femoral Neck Medial Left"},
        11: {"name": "Femoral Neck Centerline Left"},
    }

    detector = FemurDetector(unet, img_size, channels_map)
    detector.load_model(model_path)
    return detector


def runner(detector, img_path):
    img = Image.open(img_path)
    output = detector.process(img)
    return output


if __name__ == "__main__":
    model_path = "model_v1.0.0/checkpoints/epoch=29-step=1170.ckpt"
    detector = load_model(model_path)
    img_path = "visualisations/Becken2015-PA000012_0_crop.png"
    output = runner(detector, img_path)
    img = Image.open(img_path)
    img.convert("L")
    img = np.asarray(img)

    pyplot.imshow(img, cmap="gray")
    right = output["Right"]
    channels = right.keys()
    for channel in channels:
        if right[channel] is not None:
            x1 = right[channel][0]
            y1 = right[channel][1]
            x2 = right[channel][2]
            y2 = right[channel][3]
            pyplot.plot([y1, y2], [x1, x2], color="green")
    left = output["Left"]
    channels = left.keys()
    for channel in channels:
        if left[channel] is not None:
            x1 = left[channel][0]
            y1 = left[channel][1]
            x2 = left[channel][2]
            y2 = left[channel][3]
            pyplot.plot([y1, y2], [x1, x2], color="blue")
    pyplot.axis('off')
    pyplot.savefig("visualisations/pred_Becken2015-PA000012_0_crop.png")
