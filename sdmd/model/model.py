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


class SDMDConstant(nn.Module):
    def __init__(self, nums_classes: Sequence[int]):
        super().__init__()

        self.distributions = nn.ParameterList(nn.Parameter(torch.zeros(num_classes)) for num_classes in nums_classes)
    
    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        shape_prefix = x.shape[:-3]
        return tuple(F.log_softmax(distribution, dim=0).expand(*shape_prefix, -1) for distribution in self.distributions)


class SDMDLinear(nn.Module):
    def __init__(self, nums_classes: Sequence[int], size_input: int):
        super().__init__()

        self.classifier_heads = nn.ModuleList(nn.Sequential(
            nn.Linear(size_input, num_classes),
            nn.LogSoftmax(dim=-1),
        ) for num_classes in nums_classes)
    
    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = torch.flatten(x, 1)
        x = tuple(classifier_head(x) for classifier_head in self.classifier_heads)
        return x


class SDMDMultilayerPerceptron(nn.Module):
    def __init__(self, nums_classes: Sequence[int], size_hidden: int, size_input: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(size_input, size_hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier_heads = nn.ModuleList(nn.Sequential(
            nn.Linear(size_hidden, num_classes),
            nn.LogSoftmax(),
        ) for num_classes in nums_classes)
    
    def forward_extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.features(x)
        return x

    def forward_classify(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = tuple(classifier_head(x) for classifier_head in self.classifier_heads)
        return x

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.forward_extract_features(x)
        x = self.forward_classify(x)
        return x


class SDMDAlexNet(nn.Module):
    def __init__(self, nums_classes: Sequence[int], size_hidden: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.preclassifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, size_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(size_hidden, size_hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier_heads = nn.ModuleList(nn.Sequential(
            nn.Linear(size_hidden, num_classes),
            nn.LogSoftmax(dim=-1),
        ) for num_classes in nums_classes)

    def forward_extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x
    
    def forward_classify(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = torch.flatten(x, 1)
        x = self.preclassifier(x)
        x = tuple(classifier_head(x) for classifier_head in self.classifier_heads)
        return x

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.forward_extract_features(x)
        x = self.forward_classify(x)
        return x


class SDMDVGG11BNNet(nn.Module):
    def __init__(self, nums_classes: Sequence[int], size_hidden: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.preclassifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, size_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(size_hidden, size_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.classifier_heads = nn.ModuleList(nn.Sequential(
            nn.Linear(size_hidden, num_classes),
            nn.LogSoftmax(),
        ) for num_classes in nums_classes)

    def forward_extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x
    
    def forward_classify(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = torch.flatten(x, 1)
        x = self.preclassifier(x)
        x = tuple(classifier_head(x) for classifier_head in self.classifier_heads)
        return x
    
    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.forward_extract_features(x)
        x = self.forward_classify(x)
        return x