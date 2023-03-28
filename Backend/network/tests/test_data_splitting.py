"""
@description: Unit tests to test data splitting and test information leakage between splits

@TODO
Test if working. Can be outdated.

@author: Muhammad Abdullah
"""
import os
import unittest

import numpy as np
from torch.utils.data import Subset

from Backend.network.dataloader.dataloader import Dataset
from Backend.network.utils.data_splitting import k_fold_cross_validation, random_split_dataset


class DataSplitting(unittest.TestCase):
    def setUp(self):
        self.ROOT_DIR = "/proj/ciptmp/ic33axaq/IIML/data/"
        self.dataset_path = "train_img_labels_paths.csv"
        self.dataset = Dataset(os.path.join(self.ROOT_DIR, self.dataset_path))
        self.patient_ids = list(map(lambda x: x["patient_id"], self.dataset.data))

    def test_random_splitting(self):
        # Random split can leak training data to val and test datasets.
        # Because each patient have multiple images, images of same patients
        # can go into train, val, and test datasets.
        train, val, test = random_split_dataset(self.dataset)
        print("####### Random Splitting ####")
        print("test Images ", len(test))
        print("val Images ", len(val))
        print("train Images ", len(train))

    def test_k_fold_cross_validation(self):
        print("##### K-fold Cross Validation ####")
        assert len(self.patient_ids) == len(self.dataset) # duplicate ids will be removed in next function call
        k = 5
        folds = k_fold_cross_validation(k, self.patient_ids)
        test_fold = folds[0]
        folds = folds[1:]
        val_fold = folds[0]
        train_fold = np.concatenate(folds[1:])
        assert len(test_fold) == len(val_fold)
        test_idx = list(filter(lambda x: x["patient_id"] in test_fold, self.dataset.data))
        test_idx = list(map(lambda x: x["id"], test_idx))
        val_idx = list(filter(lambda x: x["patient_id"] in val_fold, self.dataset.data))
        val_idx = list(map(lambda x: x["id"], val_idx))
        train_idx = list(filter(lambda x: x["patient_id"] in train_fold, self.dataset.data))
        train_idx = list(map(lambda x: x["id"], train_idx))
        print("Patients in test fold ", len(test_fold), "Images in test ", len(test_idx))
        print("Patients in val fold ", len(val_fold), "Images in val ", len(val_idx))
        print("Patients in train fold ", len(train_fold), "Images in train ", len(train_idx))
        print("Total patients ", len(test_fold) + len(val_fold) + len(train_fold))
        print("Total Images ", len(test_idx) + len(val_idx) + len(train_idx))
        test_loader = Subset(self.dataset, test_idx)
        val_loader = Subset(self.dataset, val_idx)
        train_loader = Subset(self.dataset, train_idx)
        assert len(train_loader) == len(train_idx)
        assert len(val_loader) == len(val_idx)
        assert len(test_loader) == len(test_idx)
