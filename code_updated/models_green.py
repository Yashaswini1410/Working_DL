"""
models_green.py  -  Task 2 (Green Initiative)

A downscaled "green" model kept SEPARATE from the Task 1 models.
It reuses the existing, already-correct ResBlock from models.py, so there is
no duplicated block logic - only the smaller network is defined here.
"""
import torch
import torch.nn as nn

from models import ResBlock   # reuse the Task 1 residual block


class GreenResNet(nn.Module):
    """Green (downscaled) ResNet: same design as ResNet18 but HALF width.

    Channels [32, 64, 128, 256] instead of [64, 128, 256, 512] -> ~4x fewer
    parameters, ~half the compute, accuracy usually within a couple of points.
    Reducing width = reducing model capacity (Session 7) while keeping the
    residual/skip connections (Session 9) so the smaller net still trains well.
    """
    def __init__(self, in_channels, num_classes, width=32, **kwargs):
        super().__init__()
        activation = getattr(nn, kwargs.get("activation_str") or "ReLU")

        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8   # 32, 64, 128, 256

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.activation = activation(inplace=True)

        self.stage1 = nn.Sequential(
            ResBlock(c1, c1, activation(inplace=True), stride=1),
            ResBlock(c1, c1, activation(inplace=True), stride=1)
        )
        self.stage2 = nn.Sequential(
            ResBlock(c1, c2, activation(inplace=True), stride=2),
            ResBlock(c2, c2, activation(inplace=True), stride=1)
        )
        self.stage3 = nn.Sequential(
            ResBlock(c2, c3, activation(inplace=True), stride=2),
            ResBlock(c3, c3, activation(inplace=True), stride=1)
        )
        self.stage4 = nn.Sequential(
            ResBlock(c3, c4, activation(inplace=True), stride=2),
            ResBlock(c4, c4, activation(inplace=True), stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c4, num_classes)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)