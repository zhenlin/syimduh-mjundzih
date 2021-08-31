import torch
import torch.nn as nn
import torch.nn.functional as F

from . import nn_init

from typing import Sequence, Tuple


class SDMDAlexNetEncoder(nn.Module):
    def __init__(self):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x


class SDMDAlexNetDecoder(nn.Module):
    def __init__(self, size_target: int):
        super().__init__()

        size_0 = size_target
        size_1 = (size_0 + 2 * 2 - 11) // 4 + 1
        size_2 = (size_1 - 3) // 2 + 1
        size_3 = (size_2 - 3) // 2 + 1
        size_4 = (size_3 - 3) // 2 + 1

        unpool_stride = (size_4 + 5) // 6
        unpool_kernel_size = unpool_stride * 2 + 1
        unpool_padding = (5 * unpool_stride + unpool_kernel_size - size_4) // 2
        unpool_output_padding = size_4 - (5 * unpool_kernel_size - 2 * unpool_padding + unpool_kernel_size)

        output_padding = size_target - ((size_1 - 1) * 4 - 2 * 2 + 11)

        self.unpool = nn.Sequential(
            nn.Dropout2d(),
            nn.ConvTranspose2d(256, 256, kernel_size=unpool_kernel_size, stride=unpool_stride, padding=unpool_padding, output_padding=unpool_output_padding),
            nn.ReLU(inplace=True),
        )
        self.decode = nn.Sequential(
            nn.Upsample((size_3, size_3)),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample((size_2, size_2)),
            nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample((size_1, size_1)),
            nn.ConvTranspose2d(64, 1, kernel_size=11, stride=4, padding=2, output_padding=output_padding),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decode(x)
        return x


class SDMDAlexNetClassifier(nn.Module):
    def __init__(self, nums_classes: Sequence[int], size_hidden: int):
        super().__init__()

        dense_layer_1 = nn.Linear(256 * 6 * 6, size_hidden)
        nn.init.eye_(dense_layer_1.weight)
        nn.init.zeros_(dense_layer_1.bias)

        dense_layer_2 = nn.Linear(size_hidden, size_hidden)
        nn.init.eye_(dense_layer_2.weight)
        nn.init.zeros_(dense_layer_2.bias)
        
        def dense_layer_3(num_classes: int):
            layer = nn.Linear(size_hidden, num_classes)
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
            return layer

        self.preclassifier = nn.Sequential(
            nn.Dropout2d(),
            nn.Flatten(),
            dense_layer_1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            dense_layer_2,
            nn.ReLU(inplace=True),
        )
        self.classifier_heads = nn.ModuleList(nn.Sequential(
            dense_layer_3(num_classes),
            nn.LogSoftmax(dim=-1),
        ) for num_classes in nums_classes)
   
    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.preclassifier(x)
        x = tuple(classifier_head(x) for classifier_head in self.classifier_heads)
        return x


class SDMDAlexNet(nn.Module):
    def __init__(self, nums_classes: Sequence[int], size_hidden: int):
        super().__init__()

        self.encoder = SDMDAlexNetEncoder()
        self.classifier = SDMDAlexNetClassifier(nums_classes, size_hidden)
    
    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.encoder(x)
        x = self.classifier(x)
        return x