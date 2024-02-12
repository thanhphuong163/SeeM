# %%
import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

# %%
data_path = "../../raw_datasets/heart/processed.cleveland.data"
columns = [
    "age",
    "sex",
 	"cp",
 	"trestbps",
	"chol",
 	"fbs",
	"restecg",
 	"thalach",
 	"exang",
 	"oldpeak",
	"slope",
 	"ca",
 	"thal",
 	"label",
]
df = pd.read_csv(data_path, header=None, names=columns)
print(df.info())
# %%
print(df["ca"].unique())
print(df['thal'].unique())
df.replace('?', np.nan, inplace=True)
df.dropna(axis=0, inplace=True)
df["ca"] = df["ca"].astype(float)
df["thal"] = df["thal"].astype(float)
df.info()
# %%
features = [
    "age",
    "sex",
 	"cp",
 	"trestbps",
	"chol",
 	"fbs",
	"restecg",
 	"thalach",
 	"exang",
 	"oldpeak",
	"slope",
 	"ca",
 	"thal",
]
data = df[features]
label = df[['label']]
data.to_parquet(
    path="../../../raw_datasets/heart/data.parquet",
    index=False,
    engine="fastparquet")
label.to_parquet(
    path="../../../raw_datasets/heart/label.parquet",
    index=False,
    engine="fastparquet")
# %%
