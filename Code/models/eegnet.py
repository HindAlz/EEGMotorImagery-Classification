import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes=4,
        n_channels=22,
        n_samples=1001,
        dropout=0.5,
        kernel_length=32,
        F1=8,
        D=2,
        F2=16,
    ):
        super().__init__()

        spatial_filters = F1 * D

        self.conv_temporal = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, kernel_length),
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            F1,
            eps=1e-3,
            momentum=0.01,
        )

        self.conv_spatial = nn.Conv2d(
            F1,
            spatial_filters,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            spatial_filters,
            eps=1e-3,
            momentum=0.01,
        )
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depthwise = nn.Conv2d(
            spatial_filters,
            spatial_filters,
            kernel_size=(1, 16),
            padding="same",
            groups=spatial_filters,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(
            spatial_filters,
            F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(
            F2,
            eps=1e-3,
            momentum=0.01,
        )
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        flattened_size = F2 * (n_samples // 4 // 8)
        self.classifier = nn.Linear(flattened_size, n_classes)

    def _features(self, x):
        x = self.conv_temporal(x)
        x = self.bn1(x)

        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x

    def forward(self, x):
        features = self._features(x)
        return self.classifier(features.flatten(start_dim=1))