import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_classes=4,
        n_channels=22,
        n_samples=1001,
        dropout=0.5,
    ):
        super().__init__()

        self.conv_time = nn.Conv2d(
            1,
            40,
            kernel_size=(1, 13),
            bias=True,
        )
        self.conv_spatial = nn.Conv2d(
            40,
            40,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(
            40,
            eps=1e-5,
            momentum=0.1,
        )
        self.pool = nn.AvgPool2d(
            kernel_size=(1, 35),
            stride=(1, 7),
        )
        self.dropout = nn.Dropout(dropout)

        temporal_size = n_samples - 13 + 1
        pooled_size = (temporal_size - 35) // 7 + 1
        flattened_size = 40 * pooled_size

        self.classifier = nn.Linear(
            flattened_size,
            n_classes,
        )

    def _features(self, x):
        x = self.conv_time(x)
        x = self.conv_spatial(x)
        x = self.bn(x)

        x = x.square()
        x = self.pool(x)
        x = torch.clamp(x, min=1e-7, max=1e4)
        x = torch.log(x)
        x = self.dropout(x)

        return x.flatten(start_dim=1)

    def forward(self, x):
        features = self._features(x)
        return self.classifier(features)