"""Our custom ResNet architectures. Both are trained from scratch."""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CustomResNet(nn.Module):
    """V1 and V2 share the same network; their training methods differ."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 48, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ResidualBlock(48, 48), ResidualBlock(48, 48),
            ResidualBlock(48, 96, stride=2), ResidualBlock(96, 96),
            ResidualBlock(96, 192, stride=2), ResidualBlock(192, 192),
            ResidualBlock(192, 256, stride=2), ResidualBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return self.fc(torch.flatten(self.pool(x), 1))


class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + shortcut


class ResNetV3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 48, 3, padding=1, bias=False)
        self.layers = nn.Sequential(
            PreActBlock(48, 96), PreActBlock(96, 96), PreActBlock(96, 96),
            PreActBlock(96, 144, stride=2), PreActBlock(144, 144), PreActBlock(144, 144),
            PreActBlock(144, 192, stride=2), PreActBlock(192, 192), PreActBlock(192, 192))
        self.bn = nn.BatchNorm2d(192)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.relu(self.bn(x))
        return self.fc(torch.flatten(self.pool(x), 1))


def make_model(version):
    if version in ("v1", "v2"):
        return CustomResNet()
    if version == "v3":
        return ResNetV3()
    raise ValueError("Version must be v1, v2, or v3")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
