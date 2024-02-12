import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")


def nll_score_fn(X, mu_priors, std_priors):
    nll_scores = 0
    nll_fn = nn.GaussianNLLLoss(eps=1e-12, reduction="none")
    for Xv, mu_prior, std_prior in zip(X, mu_priors, std_priors):
        nll_scores += nll_fn(Xv, mu_prior, std_prior**2).sum(dim=-1)
    return nll_scores


def nll_L_score_fn(X, mus_priors, stds_priors):
    nll_scores = 0
    for mu_priors, std_priors in zip(mus_priors, stds_priors):
        nll_fn = nn.GaussianNLLLoss(eps=1e-12, reduction="none")
        for Xv, mu_prior, std_prior in zip(X, mu_priors, std_priors):
            nll_scores += nll_fn(Xv, mu_prior, std_prior**2).sum(dim=-1)
    return nll_scores / len(mus_priors)


def pij(X, k):
    # X: (N, dx)
    # return (N, N-1)
    N = X.size(0)
    dist_table = []
    for i, x_i in enumerate(X):
        row = []
        for j, x_j in enumerate(X):
            row.append(torch.sum((x_i - x_j) ** 2))
        row_ts = torch.stack(row)
        dist_table.append(row_ts)
    dist_ts = torch.stack(dist_table)
    # print(dist_ts.size())
    pX = []
    for i in range(N):
        row = torch.zeros((N,), device=DEVICE) + 1e-24
        topK_dist_indices = torch.sort(dist_ts[i], descending=False)[1][1 : k + 1]  # get index
        for j in topK_dist_indices:
            # for j in range(N):
            # denominator = torch.sum(torch.stack([torch.exp(-dist_ts[i, k]) for k in range(N) if k != i]))
            denominator = torch.sum(
                torch.exp(-dist_ts[i, topK_dist_indices]) + 1e-24
            )  # - torch.exp(-dist_ts[i, i])
            row[j] = torch.exp(-dist_ts[i, j]) / denominator
        pX.append(row)
    pX_ts = torch.stack(pX, dim=0)
    # print(pX_ts.size())
    return pX_ts

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
            # for j in range(N):
            # denominator = torch.sum(torch.stack([torch.exp(-dist_ts[i, k]) for k in range(N) if k != i]))
            denominator = torch.sum(
                torch.exp(-dist_ts[i, topK_dist_indices]) + 1e-24
            )  # - torch.exp(-dist_ts[i, i])
            row[j] = torch.exp(-dist_ts[i, j]) / denominator
            
        pX.append(row)
    pX_ts = torch.stack(pX, dim=0)
    # print(pX_ts)
    return pX_ts + 1e-24


def KL_PQ(pX, qZ):
    N = pX.size(0)
    scores = []
    
    score_ = torch.sum(pX*torch.log(pX / qZ), dim=1)
    
    for i in range(N):
        # score = 0
        # for j in range(N):
        #     if j != i:
        #         score += pX[i, j] * torch.log(pX[i, j] / qZ[i, j])
        # scores.append(score)
        score_[i] -= pX[i,i] * torch.log(pX[i,i] / qZ[i,i])
        # if i%100 == 0:
        #     # print(score)
        #     print(score_[i])
        
    return score_
    # return torch.stack(scores, dim=0)


def KL_weighted_PQ(pX, qZ, nll):
    N = pX.size(0)
    scores = []
    for i in range(N):
        score = 0
        for j in range(N):
            if j != i:
                score += (nll[j] / (torch.sum(nll) - nll[i])) * pX[i, j] * torch.log(pX[i, j] / qZ[i, j])
        scores.append(score)
    return torch.stack(scores, dim=0)


def sne_score_fn(X, Z, k=30):
    # X: (V, N, dv)
    # Z: (L, N, dz)
    Z = Z[0]
    pZ = pij_fast(Z, k)
    # pZ = pij(Z, k)
    scores_1kl = 0
    scores_2kl = 0
    for v, Xv in enumerate(X):
        pXv = pij_fast(Xv, k)
        # pXv = pij(Xv, k)
        scores_1kl += KL_PQ(pZ, pXv)
        scores_2kl += scores_1kl + KL_PQ(pXv, pZ)
    mask = torch.isfinite(scores_1kl)
    # print(scores_1kl)
    # print(pZ[mask==False])
    # print(pXv[mask==False])
    return {
        'sne_1kl_scores': scores_1kl,
        'sne_2kl_scores': scores_2kl
    }


def weighted_sne_score_fn(X, Z, nll, k=30):
    # X: (V, N, dv)
    # Z: (L, N, dz)
    Z = Z[0]
    pZ = pij(Z, k)
    scores = 0
    for v, Xv in enumerate(X):
        pXv = pij(Xv, k)
        scores += KL_weighted_PQ(pZ, pXv, nll)
        # scores += KL_PQ(pXv, pZ)
    return scores
