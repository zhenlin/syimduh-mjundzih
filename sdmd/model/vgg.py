import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Tuple


class SDMDVGG11BN(nn.Module):
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