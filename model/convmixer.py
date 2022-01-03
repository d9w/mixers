import torch
import torch.nn as nn
from .mlp_mixer import Classifier


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):

    def __init__(self, dim, kernel_size):
        super().__init__()
        self.model = nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConvMixer(nn.Module):

    def __init__(self, num_classes: int,
                 hidden_dim: int = 512,
                 patch_size: int = 4,
                 kernel_size: int = 8,
                 depth: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(3, hidden_dim, kernel_size = patch_size, stride=patch_size)
        self.patchbn = nn.BatchNorm2d(hidden_dim)
        layers = [ConvMixerBlock(hidden_dim, kernel_size) for _ in range(depth)]
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.patchbn(x)
        x = self.layers(x)
        x = self.pool(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x
