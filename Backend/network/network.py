"""
@description: It contains UNet class which is pytorch lightning based wrapper.
For docs related to pytorch lightning, see https://pytorch-lightning.readthedocs.io/en/stable/
@author: Muhammad Abdullah
"""
import copy

import numpy as np
from pytorch_lightning import LightningModule
from sklearn import linear_model
import torch
import torch.nn.functional as F
import torch.optim

from Backend.network.utils.math_funcs import get_centroid_or_argmax_of_2d_vector


class UNetProd(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.unet = model

    def forward(self, x):
        x = self.unet(x)
        return x


class UNet(LightningModule):
    def __init__(self, model, type, lr=1e-3, loss=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.unet = model
        self.lr = lr
        self.type = type
        self.loss = loss
        self.channels = {
            2: {"name": "shaft_rc"},
            5: {"name": "shaft_lc"},
            8: {"name": "neck_rc"},
            11: {"name": "neck_lc"},
        }


    def forward(self, x):
        x = self.unet(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x = train_batch["image"]
        target = train_batch[self.type]
        pred = self(x)
        loss = self.loss(pred, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["image"]
        target = val_batch[self.type]
        pred = self(x)
        loss = self.loss(pred, target)
        self.log("val_loss", loss)
        val_acc_crit = torch.mean(0.5*torch.abs((pred+1)-(target+1)))
        self.log("val_loss_crit", val_acc_crit)
        ### Network Evaluation
        pred = torch.sigmoid(pred)
        pred[pred<0.95] = 0
        target = torch.sigmoid(target)
        target[target<0.5] = 0
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        batch = pred.shape[0]
        target_points = val_batch["points"]
        target_points = target_points.detach().cpu().numpy()
        # Ransac
        batch_angular_error = copy.deepcopy(self.channels)
        for i in range(batch):
            for j in self.channels.keys():
                single_channel_pred_mask = pred[i][j]
                single_channel_gt_mask = target[i][j]
                if np.max(single_channel_gt_mask) <= 0.5:
                    continue
                ransac_x, ransac_y = np.where(single_channel_pred_mask>0)
                if len(ransac_x) < 2:
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
                points = target_points[i][j:j+2].reshape(-1)
                m1 = self.get_slope(points)
                m2 = self.get_slope([y1, x1, y2, x2])
                angle = self.get_angle_using_slopes(m1, m2)
                if "angular_error" in batch_angular_error[j].keys():
                    batch_angular_error[j]["angular_error"].append(angle)
                else:
                    batch_angular_error[j]["angular_error"] = []
                    batch_angular_error[j]["angular_error"].append(angle)
                centroid_p = get_centroid_or_argmax_of_2d_vector(single_channel_pred_mask, mode="centroid")
                centroid_p = np.ceil(np.flip(centroid_p))
                centroid_g = get_centroid_or_argmax_of_2d_vector(single_channel_gt_mask, mode="centroid")
                centroid_g = np.flip(centroid_g)
                dist = np.linalg.norm(centroid_p-centroid_g)
                if "error_dist" in batch_angular_error[j].keys():
                    batch_angular_error[j]["error_dist"].append(dist)
                else:
                    batch_angular_error[j]["error_dist"] = []
                    batch_angular_error[j]["error_dist"].append(dist)


        for j in self.channels.keys():
            if "angular_error" in batch_angular_error[j].keys() and len(batch_angular_error[j]["angular_error"]) > 0:
                mean_angular_error = sum(batch_angular_error[j]["angular_error"])/len(batch_angular_error[j]["angular_error"])
                mean_angular_error = 180 - mean_angular_error
                self.log(f"val_angular_error_{batch_angular_error[j]['name']}", mean_angular_error, on_step=False, on_epoch=True)
            if "error_dist" in batch_angular_error[j].keys() and len(batch_angular_error[j]["error_dist"]) > 0:
                mean_error_dist = sum(batch_angular_error[j]["error_dist"])/len(batch_angular_error[j]["error_dist"])
                self.log(f"val_error_dist_{batch_angular_error[j]['name']}", mean_error_dist, on_step=False, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        x = test_batch["image"]
        target = test_batch[self.type]
        pred = self(x)
        loss = self.loss(pred, target)
        self.log("test_loss", loss)
        val_acc_crit = torch.mean(0.5*torch.abs((pred+1)-(target+1)))
        self.log("test_loss_crit", val_acc_crit)
        ### Network Evaluation
        pred = torch.sigmoid(pred)
        pred[pred<0.95] = 0
        target = torch.sigmoid(target)
        target[target<0.5] = 0
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        batch = pred.shape[0]
        target_points = test_batch["points"]
        target_points = target_points.detach().cpu().numpy()
        # Ransac
        batch_angular_error = copy.deepcopy(self.channels)
        for i in range(batch):
            for j in self.channels.keys():
                single_channel_pred_mask = pred[i][j]
                single_channel_gt_mask = target[i][j]
                if np.max(single_channel_gt_mask) <= 0.5:
                    continue
                ransac_x, ransac_y = np.where(single_channel_pred_mask>0)
                if len(ransac_x) < 2:
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
                points = target_points[i][j:j+2].reshape(-1)
                m1 = self.get_slope(points)
                m2 = self.get_slope([y1, x1, y2, x2])
                angle = self.get_angle_using_slopes(m1, m2)
                if "angular_error" in batch_angular_error[j].keys():
                    batch_angular_error[j]["angular_error"].append(angle)
                else:
                    batch_angular_error[j]["angular_error"] = []
                    batch_angular_error[j]["angular_error"].append(angle)
                centroid_p = get_centroid_or_argmax_of_2d_vector(single_channel_pred_mask, mode="centroid")
                centroid_p = np.ceil(np.flip(centroid_p))
                centroid_g = get_centroid_or_argmax_of_2d_vector(single_channel_gt_mask, mode="centroid")
                centroid_g = np.flip(centroid_g)
                dist = np.linalg.norm(centroid_p-centroid_g)
                if "error_dist" in batch_angular_error[j].keys():
                    batch_angular_error[j]["error_dist"].append(dist)
                else:
                    batch_angular_error[j]["error_dist"] = []
                    batch_angular_error[j]["error_dist"].append(dist)


        for j in self.channels.keys():
            if "angular_error" in batch_angular_error[j].keys() and len(batch_angular_error[j]["angular_error"]) > 0:
                mean_angular_error = sum(batch_angular_error[j]["angular_error"])/len(batch_angular_error[j]["angular_error"])
                mean_angular_error = 180 - mean_angular_error
                self.log(f"test_angular_error_{batch_angular_error[j]['name']}", mean_angular_error)
            if "error_dist" in batch_angular_error[j].keys() and len(batch_angular_error[j]["error_dist"]) > 0:
                mean_error_dist = sum(batch_angular_error[j]["error_dist"])/len(batch_angular_error[j]["error_dist"])
                self.log(f"test_error_dist_{batch_angular_error[j]['name']}", mean_error_dist)

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        y = self(x)
        return y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_slope(self, points):
        x1, y1, x2, y2 = points
        slope = (y2 - y1)/(x2 - x1 + np.finfo(float).eps)
        return slope
    
    def get_angle_using_slopes(self, m1, m2):
        if isinstance(m1, np.ndarray):
            m1 = m1[0]
            m2 = m2[0]
        angle = np.abs(np.arctan((m1 - m2)/(1 + m1*m2)))
        angle = np.rad2deg(angle)
        angle = 180 - angle
        angle = np.rint(angle)
        return angle
