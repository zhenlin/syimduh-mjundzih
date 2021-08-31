import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Tuple


class MultiTop1Loss(nn.Module):
    reduction: str

    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        self.reduction = reduction

    def forward(self, input: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        losses = []

        reduction = self.reduction

        for group_idx, group_input in enumerate(input):
            pred = group_input.topk(1).indices

            losses.append(1 - (pred == target[..., group_idx, None]).to(dtype=torch.int32))

        loss_vec = sum(losses)

        if reduction == 'none':
            return loss_vec
        elif reduction == 'mean':
            return torch.mean(loss_vec)
        elif reduction == 'sum':
            return torch.sum(loss_vec)
        else:
            raise ValueError(f'{reduction} is not a valid value for reduction')


class MultiNLLLoss(nn.Module):
    ignore_index: int
    reduction: str

    def __init__(self, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()

        self.ignore_index = int(ignore_index)
        self.reduction = reduction

    def forward(self, input: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        losses = []

        ignore_index = self.ignore_index
        reduction = self.reduction

        for group_idx, group_input in enumerate(input):
            losses.append(F.nll_loss(group_input, target[..., group_idx], ignore_index=ignore_index, reduction=reduction))

        return sum(losses)


def mse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    x = (input - target).flatten(1)
    loss_vec = (x ** 2).sum(1)

    if reduction == 'none':
        return loss_vec
    elif reduction == 'mean':
        return torch.mean(loss_vec)
    elif reduction == 'sum':
        return torch.sum(loss_vec)
    else:
        raise ValueError(f'{reduction} is not a valid value for reduction')


class MSELoss(nn.Module):
    reduction: str

    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mse_loss(input, target, self.reduction)


def binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    loss_vec = F.binary_cross_entropy(input, target, reduction='none').flatten(1).sum(1)

    if reduction == 'none':
        return loss_vec
    elif reduction == 'mean':
        return torch.mean(loss_vec)
    elif reduction == 'sum':
        return torch.sum(loss_vec)
    else:
        raise ValueError(f'{reduction} is not a valid value for reduction')


class BCELoss(nn.Module):
    reduction: str

    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_cross_entropy(input, target, self.reduction)


class MultiScale2dBCELoss(nn.Module):
    scales: Sequence[int]
    reduction: str

    def __init__(self, scales: Sequence[int], reduction: str = 'mean'):
        super().__init__()

        self.scales = tuple(int(s) for s in scales)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []

        reduction = self.reduction

        for scale in self.scales:
            losses.append(binary_cross_entropy(F.avg_pool2d(input, kernel_size=scale), F.avg_pool2d(target, kernel_size=scale), reduction='none'))
        
        loss_vec = sum(losses)

        if reduction == 'none':
            return loss_vec
        elif reduction == 'mean':
            return torch.mean(loss_vec)
        elif reduction == 'sum':
            return torch.sum(loss_vec)
        else:
            raise ValueError(f'{reduction} is not a valid value for reduction')
     

