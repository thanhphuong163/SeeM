from typing import Any, Callable, Dict, List
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")


class VAE(nn.Module):
    def __init__(
        self,
        in_szes: List[int],
        hidden_sz: int,
        enc_intermediate_sz: int,
        dec_intermediate_sz: int,
        L: int,
        alpha: float,
    ):
        super(VAE, self).__init__()
        self.in_szes = in_szes
        self.latent_sz = min(in_szes) - 1
        self.hidden_sz = hidden_sz
        self.enc_intermediate_sz = enc_intermediate_sz
        self.dec_intermediate_sz = dec_intermediate_sz
        self.L = L
        self.alpha = alpha
        # Define model
        self.encoder = Encoder(
            self.in_szes,
            self.enc_intermediate_sz,
            self.hidden_sz,
            self.latent_sz,
            self.L,
        )
        self.sigma_z = nn.Parameter(
            alpha * torch.ones((1, self.latent_sz), requires_grad=False)
        )
        self.decoder = Decoder(
            self.in_szes,
            self.dec_intermediate_sz,
            self.latent_sz,
        )

    def cal_loss(self, X, out):
        p_z = Normal(torch.zeros_like(out["mu_post"]), self.sigma_z)
        q_z = Normal(out["mu_post"], out["std_post"])
        kl_loss = kl_divergence(p_z, q_z).sum(dim=-1).mean(dim=0)

        nll_loss = 0
        for mu_priors, std_priors in zip(out["mu_priors"], out["std_priors"]):
            log_likelihood_loss_fn = nn.GaussianNLLLoss()
            for Xv, mu_prior, sigma_prior in zip(X, mu_priors, std_priors):
                nll_loss += log_likelihood_loss_fn(Xv, mu_prior, sigma_prior**2)
        avg_nll_loss = nll_loss / self.L
        neg_elbo = avg_nll_loss + kl_loss
        return {
            "neg_elbo": neg_elbo,
            "kl_loss": kl_loss,
            "nll_loss": avg_nll_loss,
        }

    def forward(self, X):
        enc_out = self.encoder(X)
        dec_out = self.decoder(enc_out["z"])
        return {
            "z": enc_out["z"],
            "mu_post": enc_out["mu_post"],
            "std_post": enc_out["std_post"],
            "mu_priors": dec_out["mu_priors"],
            "std_priors": dec_out["std_priors"],
        }


class Encoder(nn.Module):
    def __init__(
        self,
        in_szes: List[int],
        intermediate_sz: int,
        hidden_sz: int,
        latent_sz: int,
        L: int,
    ):
        super(Encoder, self).__init__()
        self.in_szes = in_szes
        self.intermediate_sz = intermediate_sz
        self.hidden_sz = hidden_sz
        self.latent_sz = latent_sz
        self.L = L

        # Define model
        total_sz = sum(self.in_szes)
        self.F = nn.Sequential(
            nn.Linear(total_sz, self.intermediate_sz),
            nn.Tanh(),
            nn.Linear(self.intermediate_sz, self.intermediate_sz),
            nn.Tanh(),
            nn.Linear(self.intermediate_sz, self.hidden_sz),
            nn.Tanh(),
        )
        self.mu_net = nn.Sequential(nn.Linear(self.hidden_sz, self.latent_sz))
        self.sigma_net = nn.Sequential(
            nn.Linear(self.hidden_sz, self.latent_sz), nn.Softplus()
        )

    @staticmethod
    def reparameterize(mu, std):
        eps = torch.randn_like(mu)
        z = mu + std * eps
        return z

    def forward(self, X):
        X_concat = torch.concat(X, dim=-1)
        h_combined = self.F(X_concat)
        mu = self.mu_net(h_combined)
        std = self.sigma_net(h_combined)
        z = torch.stack([self.reparameterize(mu, std) for _ in range(self.L)])
        return {
            "z": z,
            "mu_post": mu,
            "std_post": std,
        }


class Decoder(nn.Module):
    def __init__(
        self,
        in_szes: List[int],
        intermediate_sz: int,
        latent_sz: int,
    ):
        super(Decoder, self).__init__()
        self.in_szes = in_szes
        self.intermediate_sz = intermediate_sz
        self.latent_sz = latent_sz
        # Define model
        self.mu_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.latent_sz, self.intermediate_sz),
                    nn.ReLU(),
                    nn.Linear(self.intermediate_sz, in_sz),
                )
                for in_sz in self.in_szes
            ]
        )
        self.sigma_priors = nn.ParameterList(
            [
                nn.Parameter(0.001 * torch.ones((in_sz,)), requires_grad=False)
                for in_sz in self.in_szes
            ]
        )

    def forward(self, Zs):
        mu_priors = [[mu_net(Z) for mu_net in self.mu_nets] for Z in Zs]
        sigma_priors = [
            [sigma.repeat(Z.shape[0], 1) for sigma in self.sigma_priors] for Z in Zs
        ]
        return {
            "mu_priors": mu_priors,
            "std_priors": sigma_priors,
        }
