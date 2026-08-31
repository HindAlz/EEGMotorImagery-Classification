import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden_size = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, channels)

    def forward(self, x):
        weights = x.mean(dim=1)
        weights = self.fc1(weights)
        weights = F.elu(weights)
        weights = self.fc2(weights)
        weights = torch.sigmoid(weights).unsqueeze(1)

        return x * weights


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout,
        activation="elu",
    ):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation
        self.activation = activation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(
            out_channels,
            eps=1e-3,
            momentum=0.01,
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(
            out_channels,
            eps=1e-3,
            momentum=0.01,
        )

        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def _activate(self, x):
        if self.activation == "elu":
            return F.elu(x)

        if self.activation == "relu":
            return F.relu(x)

        raise ValueError(f"Unknown activation: {self.activation}")

    def _chomp(self, x):
        if self.padding == 0:
            return x

        return x[:, :, :-self.padding]

    def forward(self, x):
        residual = self.downsample(x)

        output = self.conv1(x)
        output = self._chomp(output)
        output = self.bn1(output)
        output = self._activate(output)
        output = self.dropout(output)

        output = self.conv2(output)
        output = self._chomp(output)
        output = self.bn2(output)
        output = self._activate(output)
        output = self.dropout(output)

        return output + residual


class TCNBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        filters=32,
        depth=2,
        kernel_size=4,
        dropout=0.3,
        activation="elu",
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TCNResidualBlock(
                    in_channels=input_dim if index == 0 else filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                    activation=activation,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        return x.permute(0, 2, 1)


class ATCNet(nn.Module):
    def __init__(
        self,
        n_classes=4,
        n_channels=22,
        n_samples=1001,
        n_windows=3,
        attention="mha",
        eegn_F1=16,
        eegn_D=2,
        eegn_kernel_size=64,
        eegn_pool_size=7,
        eegn_dropout=0.3,
        tcn_depth=2,
        tcn_kernel_size=4,
        tcn_filters=32,
        tcn_dropout=0.3,
        tcn_activation="elu",
        fuse="average",
    ):
        super().__init__()

        self.n_windows = n_windows
        self.attention = attention
        self.fuse = fuse

        spatial_filters = eegn_F1 * eegn_D

        self.conv_temporal = nn.Conv2d(
            1,
            eegn_F1,
            kernel_size=(1, eegn_kernel_size),
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            eegn_F1,
            eps=1e-3,
            momentum=0.01,
        )

        self.conv_spatial = nn.Conv2d(
            eegn_F1,
            spatial_filters,
            kernel_size=(n_channels, 1),
            groups=eegn_F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            spatial_filters,
            eps=1e-3,
            momentum=0.01,
        )
        self.pool1 = nn.AvgPool2d(
            kernel_size=(1, eegn_pool_size)
        )
        self.drop1 = nn.Dropout(eegn_dropout)

        if attention == "mha":
            self.attention_blocks = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        spatial_filters,
                        num_heads=4,
                        batch_first=True,
                    )
                    for _ in range(n_windows)
                ]
            )
        elif attention in {"se", "cbam"}:
            self.attention_blocks = nn.ModuleList(
                [
                    ChannelAttention(spatial_filters)
                    for _ in range(n_windows)
                ]
            )
        elif attention is None:
            self.attention_blocks = nn.ModuleList(
                [nn.Identity() for _ in range(n_windows)]
            )
        else:
            raise ValueError(f"Unknown attention: {attention}")

        self.tcn_blocks = nn.ModuleList(
            [
                TCNBlock(
                    input_dim=spatial_filters,
                    filters=tcn_filters,
                    depth=tcn_depth,
                    kernel_size=tcn_kernel_size,
                    dropout=tcn_dropout,
                    activation=tcn_activation,
                )
                for _ in range(n_windows)
            ]
        )

        if fuse == "average":
            self.classifiers = nn.ModuleList(
                [
                    nn.Linear(tcn_filters, n_classes)
                    for _ in range(n_windows)
                ]
            )
        elif fuse == "concat":
            self.classifier = nn.Linear(
                tcn_filters * n_windows,
                n_classes,
            )
        else:
            raise ValueError(f"Unknown fusion method: {fuse}")

    def _conv_features(self, x):
        x = self.conv_temporal(x)
        x = self.bn1(x)

        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.pool1(x)
        x = self.drop1(x)

        return x[:, :, -1, :].permute(0, 2, 1)

    def _apply_attention(self, x, window_index):
        attention_block = self.attention_blocks[window_index]

        if self.attention == "mha":
            output, _ = attention_block(
                x,
                x,
                x,
                need_weights=False,
            )
            return output

        return attention_block(x)

    def forward(self, x):
        features = self._conv_features(x)
        time_steps = features.shape[1]

        if time_steps < self.n_windows:
            raise ValueError(
                "n_windows exceeds pooled time steps"
            )

        outputs = []

        for window_index in range(self.n_windows):
            window_end = (
                time_steps
                - self.n_windows
                + window_index
                + 1
            )
            window = features[
                :, window_index:window_end, :
            ]
            window = self._apply_attention(
                window,
                window_index,
            )
            window = self.tcn_blocks[window_index](window)
            window = window[:, -1, :]

            if self.fuse == "average":
                output = self.classifiers[window_index](window)
            else:
                output = window

            outputs.append(output)

        if self.fuse == "average":
            return torch.stack(outputs).mean(dim=0)

        return self.classifier(
            torch.cat(outputs, dim=1)
        )