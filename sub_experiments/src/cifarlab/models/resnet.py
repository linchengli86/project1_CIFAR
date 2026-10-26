"""A small ResNet for 32x32 CIFAR-10, in d2l style.

Same adaptation logic as the GoogLeNet: the ImageNet stem (7x7/2 + 3x3/2 max-pool) would
erase a 32x32 image, so it is replaced by one 3x3/1 convolution and the network keeps the
full resolution until the first strided block, giving 32 -> 16 -> 8 -> 4.

    Residual(cin, cout, stride=2)  ==  3x3-BN-ReLU -> 3x3-BN -> (+ 1x1 conv) -> ReLU

The default configuration is picked to land next to the 1.01M-parameter GoogLeNet, so the
two models can be compared at a matched budget rather than a matched layer count.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .layers import conv_bn, init_weights, scale


class Residual(nn.Module):
    """d2l's residual block; `stride=2` halves the resolution and projects the shortcut."""
    def __init__(self, cin, cout, stride=1, bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=not bn)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=not bn)
        self.bn1 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        # d2l's block projects the shortcut with a bare 1x1 conv; we add BN after it, as the
        # torchvision ResNet does -- otherwise the shortcut and the BN'd main branch start
        # out at different scales and the first epoch of every stage is wasted rescaling.
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=not bn),
                                       nn.BatchNorm2d(cout) if bn else nn.Identity())

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (self.short(x) if self.short is not None else x))


class ResNet(nn.Module):
    """3x3 stem + `sum(blocks)` residual blocks over four stages, then global-pool head.

    The default (32-64-128-128 channels, two blocks per stage except the last) lands at
    1.009M parameters, i.e. next to GoogLeNet's 1.006M, while using 111 MMACs against its
    64 MMACs: 3x3 convolutions at full resolution are simply more expensive per parameter
    than Inception's 1x1-heavy branches. The last stage gets one block because at 4x4 a
    block costs 295k parameters but only 9 MMACs -- bad value under a parameter budget.
    """
    def __init__(self, num_classes=10, width=1.0, channels=(32, 64, 128, 128),
                 blocks=(2, 2, 2, 1), dropout=0.3, bn=True, init=None):
        super().__init__()
        ch = [scale(c, width) for c in channels]
        self.stem = nn.Sequential(*conv_bn(3, ch[0], 3, bn=bn))
        layers, cin = [], ch[0]
        for i, (c, n) in enumerate(zip(ch, blocks)):
            for j in range(n):
                # only the first block of a stage downsamples
                layers.append(Residual(cin, c, 2 if (j == 0 and i > 0) else 1, bn=bn))
                cin = c
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(cin, num_classes))
        if init:
            self.apply(lambda m: init_weights(m, init))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))
