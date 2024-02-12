import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch as torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_dir, stage, scalers):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.stage = stage
        path = f"{data_dir}/{stage}/view*.parquet"
        view_paths = sorted(glob.glob(path))
        if stage == 'train':
            self.data = [
                scaler.fit_transform(pd.read_parquet(path)).astype(np.float32)\
                for scaler, path in zip(scalers, view_paths)
                ]
            label_path = f"{data_dir}/{stage}/label.parquet"
            self.ground_truth = pd.read_parquet(label_path)\
                .reset_index()['is_anomaly'].values.astype(np.int32)
        # if stage in ['valid', 'test']:
        #     self.data = [
        #         scaler.transform(pd.read_parquet(path)).astype(np.float32)\
        #         for scaler, path in zip(scalers, view_paths)
        #     ]
        #     label_path = f"{data_dir}/{stage}/label.parquet"
        #     self.ground_truth = pd.read_parquet(label_path)\
        #         .reset_index()['is_anomaly'].values.astype(np.int32)

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, index):
        views = [torch.tensor(data[index]) for data in self.data]
        y = torch.tensor(self.ground_truth[index])
        return views, y


class MyDatasetSemi(Dataset):
    def __init__(self, data_dir, stage, scalers):
        super(MyDatasetSemi, self).__init__()
        self.data_dir = data_dir
        self.stage = stage
        path = f"{data_dir}/train/view*.parquet"
        view_paths = sorted(glob.glob(path))
        label_path = f"{data_dir}/train/label.parquet"
        data_views = [scaler.fit_transform(pd.read_parquet(view_path)).astype(np.float32)
                      for scaler, view_path in zip(scalers, view_paths)]
        label = pd.read_parquet(label_path)\
                .reset_index()['is_anomaly'].values.astype(np.int32)
        if stage == 'train':
            # Only train on normal data which have label 0
            self.data = [data_view[label == 0] for data_view in data_views]
            self.ground_truth = label[label == 0]
        if stage in ['valid', 'test']:
            self.data = data_views
            self.ground_truth = label

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, index):
        views = [torch.tensor(data[index]) for data in self.data]
        y = torch.tensor(self.ground_truth[index])
        return views, y


def main():
    data_dir = "../../../ready_datasets/wdbc"
    scalers = [MinMaxScaler(), MinMaxScaler()]
    train = MyDataset(data_dir, 'train', scalers)
    valid = MyDataset(data_dir, 'valid', scalers)
    print(train.__len__())
    views, y = train.__getitem__(6)
    print(y)
    print(views)
    print(valid.__len__())
    views, y = valid.__getitem__(6)
    print(y)
    print(views)


if __name__ == '__main__':
    main()