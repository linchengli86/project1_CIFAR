"""A small VGG for 32x32 CIFAR-10, in d2l style -- optionally with residual shortcuts.

d2l's VGG-11 ends in three fully connected layers of width 4096 (138M parameters);
for 32x32 we keep exactly the block structure -- `num_convs` 3x3 convolutions at one
resolution, then a 2x2 max-pool -- and replace the head with global pooling.

`residual=True` puts a shortcut around the convolutions of each block. That is the
"VGG x ResNet" combination: VGG keeps its depth (stacked 3x3 at full width, no 1x1
bottleneck), and the identity path gives the gradient a way through it.

    vgg_block(2, 3, 48)                 ==  conv3x3-BN-ReLU, conv3x3-BN-ReLU, maxpool
    vgg_block(2, 3, 48, residual=True)  ==  the same, plus x added before the ReLU
"""
import torch
from torch import nn
from torch.nn import functional as F

from .layers import conv_bn, init_weights, scale


class ResidualBlock(nn.Module):
    """`num_convs` 3x3 convs wrapped in a shortcut, then a 2x2 max-pool."""
    def __init__(self, num_convs, cin, cout, bn=True):
        super().__init__()
        layers, c = [], cin
        for _ in range(num_convs):
            layers += conv_bn(c, cout, 3, bn=bn)
            c = cout
        self.convs = nn.Sequential(*layers)
        self.short = None
        if cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=not bn),
                                       nn.BatchNorm2d(cout) if bn else nn.Identity())
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        y = self.convs(x)
        return self.pool(F.relu(y + (self.short(x) if self.short is not None else x)))


def vgg_block(num_convs, cin, cout, bn=True, residual=False):
    """d2l's VGG block: `num_convs` convolutions at one resolution, then 2x2 max-pool."""
    if residual:
        return ResidualBlock(num_convs, cin, cout, bn)
    layers = []
    for _ in range(num_convs):
        layers += conv_bn(cin, cout, 3, bn=bn)
        cin = cout
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """Four blocks (32 -> 2 px), no padding tricks, no fully connected stack.

    The default widths (32-64-128-128, three convolutions per block) spend the whole ~1M
    parameter budget on twelve full-width 3x3 convolutions. That is the contrast with the
    ResNet: without 1x1 bottlenecks, every layer is expensive, so at equal parameters VGG
    buys fewer layers per layer-width.
    """
    def __init__(self, num_classes=10, width=1.0, channels=(32, 64, 128, 128),
                 num_convs=(3, 3, 3, 3), residual=False, dropout=0.3, bn=True, init=None):
        super().__init__()
        ch = [scale(c, width) for c in channels]
        blocks, cin = [], 3
        for n, c in zip(num_convs, ch):
            blocks.append(vgg_block(n, cin, c, bn=bn, residual=residual))
            cin = c
        self.stages = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(cin, num_classes))
        if init:
            self.apply(lambda m: init_weights(m, init))

    def forward(self, x):
        return self.head(self.stages(x))
