"""
@description: This is used to evaluate the performance of trained Network
@author: Muhammad Abdullah
"""
import os

import numpy as np
from matplotlib import pyplot
from sklearn import linear_model
import torch

from Backend.network.dataloader.dataloader import Dataset
from Backend.network.network import UNetProd
from Backend.network.unet import UNet
from Backend.network.utils.math_funcs import get_centroid_or_argmax_of_2d_vector, orientation_angle_heatmap_line


ROOT_DIR = "/proj/ciptmp/ic33axaq/IIML/"
dataset = Dataset(os.path.join(ROOT_DIR, "data/projiiml-wise22/data/pred_img_labels_paths.csv"))

channels = {
    0: {"name": "shaft_rl"},
    1: {"name": "shaft_rm"},
    2: {"name": "shaft_rc"},
    3: {"name": "shaft_ll"},
    4: {"name": "shaft_lm"},
    5: {"name": "shaft_lc"},
    6: {"name": "neck_rl"},
    7: {"name": "neck_rm"},
    8: {"name": "neck_rc"},
    9: {"name": "neck_ll"},
    10: {"name": "neck_lm"},
    11: {"name": "neck_lc"},
}

def get_slope(points):
    x1, y1, x2, y2 = points
    slope = (y2 - y1)/(x2 - x1)
    return slope

def get_angle_using_slopes(m1, m2):
    if isinstance(m1, np.ndarray):
        m1 = m1[0]
        m2 = m2[0]
    angle = np.abs(np.arctan((m1 - m2)/(1 + m1*m2)))
    angle = np.rad2deg(angle)
    angle = 180 - angle
    angle = np.rint(angle)
    return angle

def eval_heatmap_line(gt, model_path, channels_map):
    MODEL_PATH = model_path
    unet = UNet(num_classes=12, in_channels=1)
    model = UNetProd(unet)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    f, axarr = pyplot.subplots(len(gt),5)
    [axis.set_axis_off() for axis in axarr.ravel()]
    cols = ["Image", "GT Mask", "Pred Mask", "Post Process.", "Angle"]
    for ax, col in zip(axarr[0], cols):
            ax.set_title(col)
    
    angular_error = []
    euc_dist = []
    channels = 12
    for item in range(len(gt)):
        gt_item = gt[item]
        gt_img = gt_item["image"]
        gt_img = torch.reshape(gt_img, (1, *gt_img.shape))
        pred_mask = model(gt_img)
        pred_mask = pred_mask.detach().numpy()[0]
        print("max", np.max(pred_mask), "min", np.min(pred_mask))
        print("mean", np.mean(pred_mask), "std", np.std(pred_mask))
        gt_mask = gt_item["femur_map"].detach().numpy()
        print("gt max", np.max(gt_mask), "gt min", np.min(gt_mask))
        pred_mask = 1/(1+np.exp(-pred_mask))
        pred_mask[pred_mask<0.95] = 0

        gt_mask = 1/(1+np.exp(-gt_mask))
        gt_mask[gt_mask<0.5] = 0

        gt_img = gt_img.detach().numpy()[0][0]
        axarr[item,0].imshow(gt_img, cmap="gray")
        axarr[item,0].imshow(np.sum(gt_mask, axis=0), alpha=0.5, cmap="gray")
        axarr[item,1].imshow(np.sum(gt_mask, axis=0), cmap="gray")
        axarr[item,2].imshow(np.sum(pred_mask, axis=0), cmap="gray")

        # Ransac
        for i in range(channels):
            single_channel_pred_mask = pred_mask[i]
            ransac_x, ransac_y = np.where(single_channel_pred_mask>0)
            if len(ransac_x) == 0:
                channels_map[i]["points"] = None
                continue
            # Robustly fit linear model with RANSAC algorithm
            ransac = linear_model.RANSACRegressor()
            ransac.fit(ransac_x.reshape(-1,1), ransac_y.reshape(-1,1))
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
            zero_img = np.zeros(single_channel_pred_mask.shape)
            axarr[item,3].imshow(zero_img, cmap="gray")
            axarr[item,3].plot([y1, y2], [x1, x2])
            channels_map[i]["points"] = [x1, y1, x2, y2]

        #Pred CCD Angle calculations
        points_shaft_rc = channels_map[2]["points"]
        points_shaft_lc = channels_map[5]["points"]
        points_neck_rc = channels_map[8]["points"]
        points_neck_lc = channels_map[11]["points"]

        if points_shaft_rc is not None and points_neck_rc is not None:
            slope_shaft_rc = get_slope(points_shaft_rc)
            slope_neck_rc = get_slope(points_neck_rc)

            angle_right = get_angle_using_slopes(slope_shaft_rc, slope_neck_rc)
            axarr[item,4].text(0, 0.2, f"Pred RCCD -> {angle_right}", fontsize="medium")

        if points_shaft_lc is not None and points_neck_lc is not None:
            slope_shaft_lc = get_slope(points_shaft_lc)
            slope_neck_lc = get_slope(points_neck_lc)

            angle_left = get_angle_using_slopes(slope_shaft_lc, slope_neck_lc)
            axarr[item,4].text(0, 0.6, f"Pred LCCD -> {angle_left}", fontsize="medium")
        
        #GT CCD Angle calculations
        gt_points = gt_item["points"].detach().numpy()
        gt_points_shaft_rc = gt_points[4:6].reshape(-1)
        gt_points_shaft_lc = gt_points[10:12].reshape(-1)
        gt_points_neck_rc = gt_points[16:18].reshape(-1)
        gt_points_neck_lc = gt_points[22:].reshape(-1)

        if np.max(gt_points_shaft_rc) > 0 and np.max(gt_points_neck_rc) > 0:
            slope_gt_shaft_rc = get_slope(gt_points_shaft_rc)
            slope_gt_neck_rc = get_slope(gt_points_neck_rc)

            gt_angle_right = get_angle_using_slopes(slope_gt_shaft_rc, slope_gt_neck_rc)
            axarr[item,4].text(0, 0.4, f"GT RCCD -> {gt_angle_right}", fontsize="medium")

        if np.max(gt_points_shaft_lc) > 0 and np.max(gt_points_neck_lc) > 0:
            slope_gt_shaft_lc = get_slope(gt_points_shaft_lc)
            slope_gt_neck_lc = get_slope(gt_points_neck_lc)

            gt_angle_left = get_angle_using_slopes(slope_gt_shaft_lc, slope_gt_neck_lc)
            axarr[item,4].text(0, 0.8, f"GT LCCD -> {gt_angle_left}", fontsize="medium")

    pyplot.savefig(f"visualisations/predicted_heatmap_line_segmentation_masks.png", dpi=300)


if __name__ =="__main__":
    dataset.transform = False
    # model_path = "/proj/ciptmp/ic33axaq/IIML/experiments/lightning_logs/version_39/checkpoints/epoch=29-step=1170.ckpt"
    model_path = "/proj/ciptmp/ic33axaq/IIML/experiments/lightning_logs/version_44/checkpoints/epoch=29-step=1170.ckpt"
    eval_heatmap_line(dataset, model_path, channels)
