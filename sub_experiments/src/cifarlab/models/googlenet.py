"""A small GoogLeNet (Inception v1) for 32x32 CIFAR-10, in d2l style.

Deviations from the ImageNet GoogLeNet in the d2l book (224x224, 6.7M params):
  * the stem keeps the 32x32 resolution (7x7/2 + two pools would erase the image);
    only three 3x3/2 max-pools remain, so 32 -> 16 -> 8 -> 4,
  * aux classifiers are dropped (BatchNorm already makes the deep stack trainable),
  * the classifier is a 352->10 linear layer after global pooling.
Budget: 1.01M params (vs 6.7M), 63.8 MMACs / 0.13 GFLOPs per image (vs ~1.4 G MACs for the 224x224 original),
44 conv layers (2 in the stem + 6 per Inception block).

    Inception(192, 64, (96, 128), (16, 32), 32)
              in, 1x1, 1x1->3x3,  1x1->5x5,  pool->1x1   # d2l's signature, kept as is
"""
import torch
from torch import nn

from .layers import conv_bn, init_weights, scale


class Inception(nn.Module):
    """Parallel 1x1 / 1x1-3x3 / 1x1-5x5 / maxpool-1x1 branches, concatenated.

    As in Inception-v3, the 3x3 and 5x5 kernels are factorised into 1x3 -> 3x1 and
    1x5 -> 5x1: the receptive field is unchanged, but a k x k kernel costs k^2 weights
    per channel pair while 1 x k + k x 1 costs 2k, so 3x3 goes 9 -> 6 and 5x5 goes 25 -> 10.
    """
    def __init__(self, cin, c1, c2, c3, c4, bn=True):
        super().__init__()
        self.b1 = nn.Sequential(*conv_bn(cin, c1, 1, bn=bn))
        self.b2 = nn.Sequential(*conv_bn(cin, c2[0], 1, bn=bn),
                                *conv_bn(c2[0], c2[1], (1, 3), bn=bn),
                                *conv_bn(c2[1], c2[1], (3, 1), bn=bn))
        self.b3 = nn.Sequential(*conv_bn(cin, c3[0], 1, bn=bn),
                                *conv_bn(c3[0], c3[1], (1, 5), bn=bn),
                                *conv_bn(c3[1], c3[1], (5, 1), bn=bn))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), *conv_bn(cin, c4, 1, bn=bn))
        self.out = c1 + c2[1] + c3[1] + c4

    def forward(self, x):
        return torch.cat((self.b1(x), self.b2(x), self.b3(x), self.b4(x)), 1)


# The seven Inception blocks as (c1, c2, c3, c4): 2 blocks at 16x16, 3 at 8x8, 2 at 4x4.
BLOCKS = [[(32, (32, 48), (8, 16), 16), (48, (48, 64), (16, 24), 24)],
          [(64, (64, 96), (16, 32), 32), (80, (80, 112), (16, 32), 32),
           (80, (80, 128), (16, 40), 40)],
          [(96, (96, 128), (24, 48), 48), (112, (112, 144), (24, 48), 48)]]


class GoogLeNet(nn.Module):
    """Stem + 3 stages of Inception blocks (7 blocks in total) + global-pool head.

    `width` scales every channel count (the capacity knob), `bn=False` drops BatchNorm,
    `dropout` sets the head dropout, `init` overrides PyTorch's default initialisation.
    """
    def __init__(self, num_classes=10, width=1.0, dropout=0.3, bn=True, init=None):
        super().__init__()

        def s(c):
            """Scale a channel count (or tuple) by `width` (cifarlab.models.layers.scale)."""
            return scale(c, width)

        self.stem = nn.Sequential(*conv_bn(3, s(64), 3, bn=bn), nn.MaxPool2d(3, 2, 1),
                                  *conv_bn(s(64), s(64), 3, bn=bn))
        cin, self.stages = s(64), nn.ModuleList()
        for i, blocks in enumerate(BLOCKS):
            layers = [nn.MaxPool2d(3, 2, 1)] if i else []  # downsample between stages
            for c1, c2, c3, c4 in blocks:
                layers.append(Inception(cin, s(c1), s(c2), s(c3), s(c4), bn))
                cin = layers[-1].out
            self.stages.append(nn.Sequential(*layers))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(cin, num_classes))
        if init:
            self.apply(lambda m: init_weights(m, init))

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)
