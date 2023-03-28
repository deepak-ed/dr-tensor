"""
@description: Some datasplitting related utility functions
@author: Muhammad Abdullah
"""
import numpy as np
import torch
from torch.utils.data import random_split

from Backend.network.dataloader.dataloader import Dataset


def random_split_dataset(dataset: Dataset):
    [train, val, test] = random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))
    return train, val, test

def k_fold_cross_validation(k: int, ids: list) -> list:
    uniq_ids = list(set(ids))
    folds = np.array_split(uniq_ids, k)
    return folds
