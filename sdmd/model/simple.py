import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Tuple


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