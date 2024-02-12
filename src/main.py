import os
import glob
import wandb
import time
import json
import hydra
import yaml
import logging
import torch as torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule, LightningModule, Callback
from pytorch_lightning.loggers import LightningLoggerBase
from omegaconf import DictConfig, OmegaConf

import utils
from typing import Any, Callable, Dict, List


DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # set seed for random number generators in pytorch, numpy and python.random
    curr_time = time.time()
    pl.seed_everything(curr_time, workers=True)

    if cfg.get("trainer").get("fast_dev_run"):
        logger = False
    else:
        log.info("Instantiating loggers...")
        logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating datamodule <{cfg.get('datamodule').get('_target_')}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.get("datamodule"))
    # datamodule.prepare_data()
    datamodule.setup()

    log.info(f"Instantiating model <{cfg.get('model').get('_target_')}>")
    model: nn.Module = hydra.utils.instantiate(cfg.get("model"))

    log.info(f"Instantiating optimizer <{cfg.optimizer._target_}>")
    optimizer: optim.Optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=model.parameters()
    )

    log.info(f"Instantiating pl module <{cfg.pl_module._target_}>")
    pl_module: LightningModule = hydra.utils.instantiate(
        cfg.pl_module, model=model, optimizer=optimizer
    )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, accelerator=DEVICE, callbacks=callbacks, logger=logger
    )

    if logger:
        log.info("Logging hyperparameters!")
        trainer.logger.log_hyperparams(
            {
                "model": cfg.get("model"),
                "optimizer": cfg.get("optimizer"),
                "trainer": cfg.get("trainer"),
                "datamodule": cfg.get("datamodule"),
            }
        )
    start_time = time.time()
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=pl_module,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
        )
    running_time = time.time() - start_time
    record = {
        "data": cfg.get('datamodule').get('data_name'),
        "sample": cfg.get('datamodule').get('sample'),
        "anomaly_rate": cfg.get('datamodule').get('anomaly_rate'),
        "running_time": running_time,
    }

    with open(f"./time.json", "r+") as f:
        file_data = json.load(f)
        file_data['time'].append(record)
        f.seek(0)
        json.dump(file_data, f, indent = 4)
        print(json.dumps(record, indent=2))

    if cfg.get("test"):
        log.info("Starting testing!")
        trainer.test(
            model=pl_module,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path if cfg.ckpt_path else "last",
        )


if __name__ == "__main__":
    # with open("params.yaml") as f:
    #     cfg = yaml.safe_load(f)
    # cfg = OmegaConf.load("params.yaml")
    main()
