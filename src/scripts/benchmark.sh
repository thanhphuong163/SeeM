#! /bin/zsh

source activate mytorch

export HYDRA_FULL_ERROR=1

python3 main.py -m \
    test=false \
    dirs=benchmark \
    dirs.data_dir='../../ready_datasets' \
    trainer=benchmark \
    logger=default \
    callbacks=dev \
    datamodule=hear-2 \
    datamodule.anomaly_rate=2,5,15,25 \
    datamodule.scaler=robust \
    pl_module=pl_L_module \
    model=vae_sample_z_L_concat_view