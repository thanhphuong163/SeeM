# %%
import os
import sys
import glob
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydra
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from omegaconf import DictConfig, OmegaConf

DEVICE = torch.device("cpu")


# %%
def split_views(df, cfg):
    views = {f"view_{i+1}": None for i in range(cfg.get("n_views"))}
    features = list(df.columns)
    random.Random(cfg.get("random_state")).shuffle(features)
    n_feat_view = len(features) / cfg.get("n_views")
    for v in range(cfg.get("n_views")):
        start = int(v * n_feat_view)
        end = int(min((v + 1) * n_feat_view, len(features)))
        views[f"view_{v+1}"] = features[start:end]
    return views


# %%
def gen_anomalies(df_views, df_label, cfg):
    # Choose indices
    size = df_views["view_1"].shape[0]
    n_indices = int(size * cfg.get("anomaly_rate") * 0.01)
    a_indices = random.Random(cfg.get("random_state")).sample(
        list(df_views["view_1"].index), k=n_indices
    )
    b_indices = []
    for i in a_indices:
        instance_label = df_label["label"][i]
        other_indices = df_label[df_label["label"] != instance_label].index
        # b_indices.append(random.Random(cfg.get('random_state')).choice(other_indices))
        b_indices.append(random.choice(other_indices))
    # Choose view
    chosen_views = chosen_view = random.Random(cfg.get("random_state")).choices(
        list(df_views.keys()), k=len(a_indices)
    )
    # Replace values
    for i, j, view in zip(a_indices, b_indices, chosen_views):
        df_views[view].loc[i] = df_views[view].loc[j].values

    # Create ground truth
    df_label["is_anomaly"] = np.zeros((size,), dtype=int)
    df_label["is_anomaly"][a_indices] = 1

    metadata = {
        "a_indices": [int(i) for i in a_indices],
        "b_indices": [int(i) for i in b_indices],
        "chosen_views": chosen_views,
    }
    return metadata


# %%
def save_data(data_set, label_df, cfg, metadata=None, stage="train"):
    output_dir = f"{cfg.get('output_dir')}/{stage}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for view, view_df in data_set.items():
        view_df.to_parquet(path=f"{output_dir}/{view}.parquet", engine="fastparquet")
    label_df.to_parquet(path=f"{output_dir}/label.parquet", engine="fastparquet")
    if metadata:
        metadata.update(cfg)
        with open(f"{output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f)


# %%
def pij_fast(X, k):
    # X: (N, dx)
    # return (N, N-1)
    N = X.size(0)
    dist_ts = torch.cdist(X, X, p=2.0)
    # print(dist_ts.size())
    pX = []
    for i in range(N):
        row = torch.zeros((N,), device=DEVICE) + 1e-24
        topK_dist_indices = torch.sort(dist_ts[i], descending=False)[1][
            1 : k + 1
        ]  # get index
        for j in topK_dist_indices:
            denominator = torch.sum(torch.exp(-dist_ts[i, topK_dist_indices]) + 1e-24)
            row[j] = torch.exp(-dist_ts[i, j]) / denominator
        pX.append(row)
    pX_ts = torch.stack(pX, dim=0)
    # print(pX_ts.size())
    return pX_ts


def KL_PQ(pX, qZ):
    N = pX.size(0)
    score_ = torch.sum(pX * torch.log(pX / qZ), dim=1)
    for i in range(N):
        score_[i] -= pX[i, i] * torch.log(pX[i, i] / qZ[i, i])
    return score_


def sne_score_fn(X, k=30):
    # X: (V, N, dv)
    scores = 0
    pX = [pij_fast(Xi, k) for Xi in X]
    for i, Xi in enumerate(X):
        for j, Xj in enumerate(X):
            if j != i:
                scores += KL_PQ(pX[i], pX[j])
    return scores


def multiview_anomaly_filter(data_views, label, k=30, removed_rate=0.01):
    # get sne_score
    # remove top removed_rate highest score
    N = data_views["view_1"].shape[0]
    X = [StandardScaler().fit_transform(Xv) for v, Xv in data_views.items()]
    X_ts = [torch.from_numpy(Xv) for Xv in X]
    sne_scores = sne_score_fn(X_ts, k)
    K = int(removed_rate * N)
    topK_indices = torch.sort(sne_scores, descending=True)[1][:K].numpy()
    data_views = {
        view: pd.DataFrame(
            data=np.delete(Xv.values, topK_indices, axis=0), columns=Xv.columns
        )
        for view, Xv in data_views.items()
    }
    label = pd.DataFrame(
        data=np.delete(label.values, topK_indices, axis=0), columns=label.columns
    )
    return data_views, label

def multiview_anomaly_minmax_filter(data_views, label, k=30, removed_rate=0.01):
    # get sne_score
    # remove top removed_rate highest score
    N = data_views["view_1"].shape[0]
    X = [StandardScaler().fit_transform(Xv) for v, Xv in data_views.items()]
    X_ts = [torch.from_numpy(Xv) for Xv in X]
    sne_scores = sne_score_fn(X_ts, k)
    scaled_sne_scores = MinMaxScaler().fit_transform(sne_scores.reshape(-1,1)).reshape(-1,)
    # K = int(removed_rate * N)
    # topK_indices = torch.sort(sne_scores, descending=True)[1][:K].numpy()
    a = scaled_sne_scores >= 0.9
    topK_indices = a.nonzero()
    print(scaled_sne_scores[topK_indices])
    data_views = {
        view: pd.DataFrame(
            data=np.delete(Xv.values, topK_indices, axis=0), columns=Xv.columns
        )
        for view, Xv in data_views.items()
    }
    label = pd.DataFrame(
        data=np.delete(label.values, topK_indices, axis=0), columns=label.columns
    )
    return data_views, label


def singleview_anomaly_filter(data_views, label, removed_rate=0.01):
    N = data_views["view_1"].shape[0]
    X = [StandardScaler().fit_transform(Xv) for v, Xv in data_views.items()]
    X_concat = np.concatenate(X, axis=1)

    lof = LocalOutlierFactor(n_jobs=-1)
    # Train
    lof.fit(X_concat)
    # Test
    y_pred = lof.fit_predict(X_concat)
    y_score = lof.negative_outlier_factor_

    K = int(removed_rate * N)
    topK_indices = np.argsort(y_score)[:K]
    data_views = {
        view: pd.DataFrame(
            data=np.delete(Xv.values, topK_indices, axis=0), columns=Xv.columns
        )
        for view, Xv in data_views.items()
    }
    label = pd.DataFrame(
        data=np.delete(label.values, topK_indices, axis=0), columns=label.columns
    )
    return data_views, label


# %%
# def main(cfg):
#     data = pd.read_parquet(f"{cfg.get('data_dir')}/data.parquet")
#     label = pd.read_parquet(f"{cfg.get('data_dir')}/label.parquet")
#     # NOTE: switch to unsupervised scenario
#     views = split_views(data, cfg)
#     # # Split train/valid/test
#     # train_data, test_data, train_label, test_label = train_test_split(
#     #     data, label,
#     #     test_size=cfg.get('test_size'),
#     #     random_state=cfg.get('random_state')
#     # )
#     # valid_data, test_data, valid_label, test_label = train_test_split(
#     #     test_data, test_label,
#     #     test_size=0.6,
#     #     random_state=cfg.get('random_state')
#     # )

#     # Split views
#     data_views = {view: data[view_feat] for view, view_feat in views.items()}
#     # train_set = {view: train_data[view_feat] for view, view_feat in views.items()}
#     # valid_set = {view: valid_data[view_feat] for view, view_feat in views.items()}
#     # test_set = {view: test_data[view_feat] for view, view_feat in views.items()}

#     # Remove anomaly in data
#     # print(data_views['view_1'].shape)
#     data_views, label = multiview_anomaly_filter(data_views, label, k=30, removed_rate=0.03)
#     # print(data_views['view_1'].shape)

#     # Generate anomalies
#     metadata = gen_anomalies(data_views, label, cfg)
#     # valid_metadata = gen_anomalies(valid_set, valid_label, cfg)
#     # test_metadata = gen_anomalies(test_set, test_label, cfg)

#     # Save to files
#     save_data(data_views, label, cfg, metadata)
#     # save_data(train_set, train_label, cfg)
#     # save_data(valid_set, valid_label, cfg, valid_metadata, stage='valid')
#     # save_data(test_set, test_label, cfg, test_metadata, stage='test')


def start_gen_anomalies(cfg):
    data = pd.read_parquet(f"{cfg.get('data_dir')}/data.parquet")
    label = pd.read_parquet(f"{cfg.get('data_dir')}/label.parquet")
    views = split_views(data, cfg)

    # Split views
    data_views = {view: data[view_feat] for view, view_feat in views.items()}
    # print(data_views['view_1'].shape)

    # Remove single-view anomalies in data
    # data_views, label = singleview_anomaly_filter(
    #     data_views, label, removed_rate=0.01 * cfg.get("removed_rate")
    # )

    # Remove multi-view anomalies in data
    data_views, label = multiview_anomaly_minmax_filter(
        data_views, label, k=30, removed_rate=0.01 * cfg.get("removed_rate")
    )
    # print(data_views['view_1'].shape)
    # Generate anomalies
    metadata = gen_anomalies(data_views, label, cfg)

    # Save to files
    save_data(data_views, label, cfg, metadata)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    n_views = cfg.n_views
    data_name = cfg.dataname
    removed_rate = cfg.removed_rate
    samples = list(range(0, 10))
    list_random_seeds = [42, 4423, 21, 653, 767, 23, 874, 32, 753, 222]
    # list_random_seeds = [23, 874, 32, 753, 222]
    for i, rs in zip(samples, list_random_seeds):
        for anomaly_rate in [2,10,15,20,25,30]:
            settings = {
                "data_dir": f"../../raw_datasets/{data_name}",
                "output_dir": f"../../ready_datasets/removed_stdscaler_{cfg.get('removed_rate')}_threshold0.9_anomaly_datasets/{data_name}-{n_views}/sample-{i+1}/anomaly_rate-{anomaly_rate}",
                "random_state": rs,
                "n_views": n_views,
                "anomaly_rate": anomaly_rate,
                "removed_rate": cfg.get("removed_rate"),
            }
            print(
                f"Generating dataset: {data_name}\tsample: {i+1}\tanomaly rate: {anomaly_rate}"
            )
            start_gen_anomalies(settings)


# %%
if __name__ == "__main__":
    main()
    # n_views = 2
    # data_names = [
    #     # 'bcw',
    #     # 'glass',
    #     # 'heart',
    #     # 'sonar',
    #     # 'svmguide2',
    #     # 'svmguide4',
    #     # 'vehicle',
    #     'vowels',
    #     'magic',
    #     'drybean',
    #     'htru2',
    #     'winequality',
    #     'electrical_grid',
    # ]
    # list_random_seeds = [42, 4423, 21, 653, 767, 23, 874, 32, 753, 222]
    # for data_name in data_names:
    #     for i, rs in enumerate(list_random_seeds):
    #         for anomaly_rate in [5, 10, 15, 20]:
    #             cfg = {
    #                 "data_dir": f"../../../raw_datasets/{data_name}",
    #                 "output_dir": f"../../../ready_datasets/removed_anomaly_datasets/{data_name}-{n_views}/sample-{i+1}/anomaly_rate-{anomaly_rate}",
    #                 "random_state": rs,
    #                 "n_views": n_views,
    #                 "anomaly_rate": anomaly_rate,
    #             }
    #             print(f"Generating dataset: {data_name}\tsample: {i+1}\tanomaly rate: {anomaly_rate}")
    #             main(cfg)
