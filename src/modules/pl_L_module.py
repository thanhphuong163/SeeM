import logging
import hydra
import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from typing import Any, Callable, Dict, List

from sklearn.metrics import roc_auc_score

from modules.score_functions import nll_L_score_fn, sne_score_fn, weighted_sne_score_fn

log = logging.getLogger(__name__)
log_dict_params = {
    "prog_bar": False,
    "logger": True,
    "on_step": False,
    "on_epoch": True,
}


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
    ):
        super(LitModel, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.valid_auc = AUROC(task='binary')
        # self.test_auc = AUROC()

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, X):
        out = self.model(X)
        return out

    def run_step(self, batch):
        X, y = batch
        out = self.forward(X)
        loss = self.model.cal_loss(X, out)
        return out, loss

    def training_step(self, batch, _):
        _, loss = self.run_step(batch)
        train_metrics = {f"train_{key}": loss[key] for key in loss.keys()}
        self.log_dict(train_metrics, **log_dict_params)
        return {"loss": loss["neg_elbo"]}

    def cal_metrics(self, batch, out, stage):
        X, y = batch
        nll_scores = nll_L_score_fn(X, out["mu_priors"], out["std_priors"])
        if stage == "valid":
            self.valid_auc.update(nll_scores, y)
            auc = self.valid_auc
        elif stage == "test":
            self.test_auc.update(nll_scores, y)
            auc = self.test_auc
        return {f"{stage}_nll_score_auc": auc}

    def validation_step(self, batch, _):
        _, y = batch
        out, loss = self.run_step(batch)
        valid_metrics = {f"valid_{key}": loss[key].detach() for key in loss.keys()}

        score_metrics = self.cal_metrics(batch, out, stage="valid")
        valid_metrics.update(score_metrics)

        self.log_dict(valid_metrics, **log_dict_params)
        return {
            "batch": batch,
            "out": out,
            # 'valid_metrics': valid_metrics
        }

    def validation_epoch_end(self, outputs):
        Xs, ys, mus_priors, stds_priors, Zs = [], [], [], [], []
        for output in outputs:
            X, y = output["batch"]
            Xs.append(X)
            ys.append(y)
            out = output["out"]
            mus_priors.append(out["mu_priors"][0])
            stds_priors.append(out["std_priors"][0])
            Zs.append(out["z"][0])
        V = len(Xs[0])
        Xs_ = [torch.cat([X[v] for X in Xs], dim=0) for v in range(V)]
        ys_ = torch.cat(ys, dim=0)
        Zs_ = torch.stack([torch.cat(Zs)], dim=0)
        # print(Xs_[0].size())
        # print(Zs_[0].size())

        # print(sne_scores.size())
        # (11, 2) : (N, V) -> (1, V)
        mus_priors_ = [
            [
                torch.cat([mu_priors[v] for mu_priors in mus_priors], dim=0)
                for v in range(V)
            ]
        ]
        stds_priors_ = [
            [
                torch.cat([std_priors[v] for std_priors in stds_priors], dim=0)
                for v in range(V)
            ]
        ]
        # print(len(mus_priors_))
        nll_scores = nll_L_score_fn(Xs_, mus_priors_, stds_priors_)
        # nll_auc = AUROC(task='binary')(nll_scores, ys_)
        nll_auc = roc_auc_score(ys_.cpu().numpy(), nll_scores.cpu().numpy())
        # print(ys_.unique(return_counts=True))
        print(f"Epoch: {self.current_epoch}")
        print(f"nll auc: {nll_auc}")
        valid_metrics = {}
        valid_metrics[f"valid_nll_score_auc"] = nll_auc
        if self.current_epoch == 99:
            # for k in [10, 20, 30, 40, 50]:
            for k in [30]:
                sne_scores = sne_score_fn(Xs_, Zs_, k)
                # print(sne_scores.size())
                # sne_1kl_auc = AUROC(task='binary')(sne_scores['sne_1kl_scores'], ys_)
                # sne_2kl_auc = AUROC(task='binary')(sne_scores['sne_2kl_scores'], ys_)
                sne_1kl_auc = roc_auc_score(ys_.cpu().numpy(), sne_scores['sne_1kl_scores'].cpu().numpy())
                # sne_2kl_auc = roc_auc_score(ys_.cpu().numpy(), sne_scores['sne_2kl_scores'].cpu().numpy())
                print(f"sne 1kl k={k} auc: {sne_1kl_auc}")
                # print(f"sne 2kl k={k} auc: {sne_2kl_auc}")
                valid_metrics[f"valid_sne_1kl_{k}_score_auc"] = sne_1kl_auc
                # valid_metrics[f"valid_sne_2kl_{k}_score_auc"] = sne_2kl_auc
                # weighted_sne_auc = AUROC()(nll_scores*sne_scores, ys_)
                # weighted_sne_scores = weighted_sne_score_fn(Xs_, Zs_, nll_scores, k)
                # weighted_sne_auc = AUROC()(weighted_sne_scores, ys_)
                # print(f"weighted sne k={k} auc: {weighted_sne_auc}")
                # valid_metrics[f"valid_weighted_sne_{k}_score_auc"] = weighted_sne_auc
                score = sne_scores['sne_1kl_scores'].cpu().numpy()
                np.savetxt('../notebooks/usecase_datasets/anomaly_score.csv', score, delimiter=',')
            self.log_dict(valid_metrics, **log_dict_params)

    def test_step(self, batch, _):
        _, y = batch
        out, loss = self.run_step(batch)
        test_metrics = {f"test_{key}": loss[key].detach() for key in loss.keys()}

        score_metrics = self.cal_metrics(batch, out, stage="test")
        test_metrics.update(score_metrics)

        self.log_dict(test_metrics, **log_dict_params)
