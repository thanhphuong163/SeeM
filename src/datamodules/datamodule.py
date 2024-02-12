import os
from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd
import hydra
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import torch
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from datamodules.dataset import MyDataset, MyDatasetSemi

# from dataset import MyDataset


N_CPU = os.cpu_count()
# n_workers = N_CPU - 2
n_workers = 1


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        sample: int,
        anomaly_rate: int,
        in_szes: List[int],
        n_views: int,
        batch_sz: int,
        scaler: str = "minmax",
    ):
        super(DataModule, self).__init__()
        self.data_dir = (
            f"{data_dir}/{data_name}/sample-{sample}/anomaly_rate-{anomaly_rate}"
        )
        self.in_szes = in_szes
        self.n_views = n_views
        self.batch_sz = batch_sz
        if scaler == "minmax":
            self.scalers = [MinMaxScaler() for _ in range(self.n_views)]
        elif scaler == "standard":
            self.scalers = [StandardScaler() for _ in range(self.n_views)]
        elif scaler == "robust":
            self.scalers = [
                RobustScaler(quantile_range=(25.0, 75.0)) for _ in range(self.n_views)
            ]
        self.num_workers = n_workers

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        # self.train_set = MyDataset(self.data_dir, "train", self.scalers)
        # self.val_set = MyDataset(self.data_dir, "train", self.scalers)
        # self.test_set = MyDataset(self.data_dir, "train", self.scalers)
        # self.predict_set = MyDataset(self.data_dir, 'test', self.scalers)

        self.train_set = MyDatasetSemi(self.data_dir, "train", self.scalers)
        print(f"training set size: {self.train_set.__len__()}")
        self.val_set = MyDatasetSemi(self.data_dir, "valid", self.scalers)
        print(f"validation set size: {self.val_set.__len__()}")
        self.test_set = MyDatasetSemi(self.data_dir, "test", self.scalers)
        print(f"test set size: {self.test_set.__len__()}")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_sz,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_sz,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_sz,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_sz,
            persistent_workers=True,
            num_workers=self.num_workers,
        )


def main():
    data_dir = "../../../ready_datasets/wdbc"
    in_szes = [4, 5]
    n_views = 2
    batch_sz = 32
    dm = DataModule(data_dir, in_szes, n_views, batch_sz, "minmax")
    dm.setup()
    for batch in dm.train_dataloader():
        X, y = batch
        print(X)
        print(y)

    for batch in dm.val_dataloader():
        X, y = batch
        print(X)
        print(y)


if __name__ == "__main__":
    main()
