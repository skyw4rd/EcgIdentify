import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if (stride == 1 and in_ch == out_ch)
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        return self.act(out)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch),
            ConvBNAct(in_ch, out_ch, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.block(x)


class EDITHNet(nn.Module):
    """
    Classification-oriented EDITH baseline:
    CNN encoder + embedding head, inspired by EDITH's CNN+similarity pipeline.
    """

    def __init__(self, in_chans=3, num_classes=1000, emb_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_chans, 32, 3, 2),
            ResidualConvBlock(32, 64, stride=2),
            ResidualConvBlock(64, 128, stride=2),
            ResidualConvBlock(128, 192, stride=2),
            ResidualConvBlock(192, 256, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.embedding(x)
        return self.classifier(x)


class ECGIoTNet(nn.Module):
    """
    Lightweight ECGIoT-style baseline:
    depthwise separable CNN focused on IoT-friendly inference.
    """

    def __init__(self, in_chans=3, num_classes=1000, width_mult=1.0):
        super().__init__()

        def c(ch):
            return max(int(ch * width_mult), 8)

        self.features = nn.Sequential(
            ConvBNAct(in_chans, c(16), 3, 2),
            DepthwiseSeparableBlock(c(16), c(24), stride=1),
            DepthwiseSeparableBlock(c(24), c(32), stride=2),
            DepthwiseSeparableBlock(c(32), c(48), stride=2),
            DepthwiseSeparableBlock(c(48), c(64), stride=2),
            DepthwiseSeparableBlock(c(64), c(96), stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c(96), num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class ECGXtractorNet(nn.Module):
    """Stronger CNN baseline with SE-residual blocks inspired by ECGXtractor-style design."""

    def __init__(self, in_chans=3, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_chans, 32, 3, 2),
            ConvBNAct(32, 32, 3, 1),
        )
        self.stage1 = nn.Sequential(
            ResidualConvBlock(32, 64, stride=2, use_se=True),
            ResidualConvBlock(64, 64, stride=1, use_se=True),
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(64, 128, stride=2, use_se=True),
            ResidualConvBlock(128, 128, stride=1, use_se=True),
        )
        self.stage3 = nn.Sequential(
            ResidualConvBlock(128, 192, stride=2, use_se=True),
            ResidualConvBlock(192, 192, stride=1, use_se=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        fused = torch.cat([avg_feat, max_feat], dim=1)
        return self.head(fused)
