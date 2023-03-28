"""
@description: This is used to train UNet using pytorch lightning Trainer class.
Tensorboard is used to log and monitor training progress.
Datasplitting is not random but it takes patient ids into account.
All images of one patient go into same split.
Online augmentation strategy is used.
https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
"""
import os

import numpy as np
from pytorch_lightning import loggers, Trainer
from torch.utils.data import DataLoader, Subset
from torch.nn import BCEWithLogitsLoss, MSELoss
import torch.cuda

from Backend.network.dataloader.dataloader import Dataset
from Backend.network.network import UNet
from Backend.network.utils.data_splitting import k_fold_cross_validation
from Backend.network.unet import UNet as myunet

ROOT_DIR = "/proj/ciptmp/ic33axaq/IIML/"
train_data_path = "train.csv"
val_data_path = "val.csv"
test_data_path = "test.csv"

train_dataset = Dataset(train_data_path)
val_dataset = Dataset(val_data_path)
test_dataset = Dataset(test_data_path)
val_dataset.transform = False
test_dataset.transform = False

epochs = 40
lr = 8e-5
optimizer = "Adam"
type_ = "femur_map"
loss = MSELoss()
num_classes = 12
in_channels = 1
precision=16
train_val_batch_size = 4

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_val_batch_size, num_workers=4)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=train_val_batch_size, num_workers=4)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=4)

# # Segmentation network trainging using UNet
model = UNet(myunet(num_classes=num_classes, in_channels=1), type_, lr=lr, loss=loss)

tensorboard = loggers.TensorBoardLogger(save_dir=ROOT_DIR + "experiments/",
                             sub_dir=type_,
                             version=None,
                             name='lightning_logs')

tensorboard.log_hyperparams({"epochs": epochs,
                            "optimizer": optimizer,
                            "lr": lr,
                            "loss": loss,
                            "in_channels": in_channels,
                            "num_classes": num_classes,
                            "precision": "fp" + str(precision),
                            "train_val_Batch_size": train_val_batch_size})

trainer = Trainer(
    precision=precision,
    amp_backend="native",
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    max_epochs = epochs,
    log_every_n_steps=25,
    logger=[tensorboard])

trainer.fit(model, train_loader, val_loader)
trainer.test(dataloaders=test_loader, ckpt_path="last")
