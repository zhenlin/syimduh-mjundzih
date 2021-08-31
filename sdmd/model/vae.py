import torch
import torch.nn as nn
import torch.nn.functional as F

from . import nn_init

import math

from typing import Literal, Sequence, Tuple


class VAEEncoder(nn.Module):
    def __init__(self, size_input: int, size_latent: int, size_hidden: int):
        super().__init__()

        self.estimator_mu = nn.Sequential(
            nn_init.Linear(size_input, size_hidden, initialization='stretched_eye_pm'),
            nn.ReLU(inplace=True),
            nn_init.Linear(size_hidden, size_latent),
        )
        
        self.estimator_log_sigma = nn.Sequential(
            nn_init.Linear(size_input, size_hidden, initialization='stretched_eye_pm'),
            nn.ReLU(inplace=True),
            nn_init.Linear(size_hidden, size_latent, initialization_scale=1e-3),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(1)
        mu = self.estimator_mu(x)
        log_sigma = self.estimator_log_sigma(x)

        return mu, log_sigma


class VAEDecoder(nn.Module):
    def __init__(self, shape_target: Tuple[int, ...], size_latent: int, size_hidden: int):
        super().__init__()

        size_target = math.prod(shape_target)
        self.size_latent = size_latent

        self.decoder = nn.Sequential(
            nn_init.Linear(size_latent, size_hidden, initialization='stretched_eye_pm'),
            nn.ReLU(inplace=True),
            nn_init.Linear(size_hidden, size_target, initialization='stretched_eye_pm'),
            nn.ReLU(inplace=True),
            nn.Unflatten(-1, shape_target),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)

        return x


def vae_get_sample(mu: torch.Tensor, log_sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = torch.exp(log_sigma)
        z = torch.randn_like(sigma)

        y = mu + sigma * z

        log_denom = math.log(2 * math.pi) / 2 * math.prod(sigma.shape[1:]) 

        prior_log_like = -torch.sum(y.flatten(1) ** 2, dim=1) / 2 - log_denom
        post_log_like = -torch.sum(z.flatten(1) ** 2, dim=1) / 2 - log_denom - torch.sum(log_sigma.flatten(1), dim=1)

        return y, prior_log_like, post_log_like


def vae_get_mode(mu: torch.Tensor, log_sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_denom = math.log(2 * math.pi) / 2 * math.prod(log_sigma.shape[1:]) 

        prior_log_like = -torch.sum(mu.flatten(1) ** 2, dim=1) / 2 - log_denom
        post_log_like = -log_denom - torch.sum(log_sigma.flatten(1), dim=1)

        return mu, prior_log_like, post_log_like


def elbo_loss(inferred: torch.Tensor, observed: torch.Tensor, prior_log_like: torch.Tensor, post_log_like: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean') -> torch.Tensor:
    log_prob = -F.binary_cross_entropy(inferred, observed, reduction='none').flatten(1).sum(1)

    elbo = log_prob + prior_log_like - post_log_like

    if reduction == 'none':
        return -elbo
    elif reduction == 'mean':
        return -torch.mean(elbo)
    elif reduction == 'sum':
        return -torch.sum(elbo)
    else:
        raise ValueError(f'{reduction} is not a valid value for reduction')


class MultiScale2dELBoLoss(nn.Module):
    scales: Sequence[int]
    reduction: str

    def __init__(self, scales: Sequence[int], reduction: str = 'mean'):
        super().__init__()

        self.scales = tuple(int(s) for s in scales)
        self.reduction = reduction

    def forward(self, inferred: torch.Tensor, observed: torch.Tensor, prior_log_like: torch.Tensor, post_log_like: torch.Tensor) -> torch.Tensor:
        losses = []

        reduction = self.reduction

        for scale in self.scales:
            losses.append(elbo_loss(F.avg_pool2d(inferred, kernel_size=scale), F.avg_pool2d(observed, kernel_size=scale), prior_log_like, post_log_like, reduction='none'))
        
        loss_vec = sum(losses)

        if reduction == 'none':
            return loss_vec
        elif reduction == 'mean':
            return torch.mean(loss_vec)
        elif reduction == 'sum':
            return torch.sum(loss_vec)
        else:
            raise ValueError(f'{reduction} is not a valid value for reduction')


def normal_kl_divergence(post_mu: torch.Tensor, post_log_sigma: torch.Tensor, prior_mu: torch.Tensor, prior_log_sigma: torch.Tensor) -> torch.Tensor:
    return prior_log_sigma - post_log_sigma + (torch.exp(2 * post_log_sigma) + (prior_mu - post_mu) ** 2) / (2 * torch.exp(2 * prior_log_sigma)) - 0.5
