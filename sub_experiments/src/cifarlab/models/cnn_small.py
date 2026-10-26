"""The two-conv CNN from the earlier exploration, ported unchanged (+ an optional BN).

Two 3x3 conv + maxpool, then a 4096->128->10 head. In its original form it has no
normalisation at all -- which is exactly why it anchors the recipe ladder: it is the
"raw pixels, nothing added" baseline. `bn=True` inserts BatchNorm after each conv so the
effect of that single trick can be measured on top.
"""
from torch import nn

from .layers import init_weights


def _conv(cin, cout, bn):
    layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1)]
    if bn:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.ReLU())
    return layers


class CNN(nn.Module):
    def __init__(self, num_classes=10, bn=False, dropout=0.0, init=None):
        super().__init__()
        self.features = nn.Sequential(
            *_conv(3, 32, bn),                      # [B, 3, 32, 32] -> [B, 32, 32, 32]
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 32, 16, 16]
            *_conv(32, 64, bn),                     # -> [B, 64, 16, 16]
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 64, 8, 8]
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        if init:
            self.apply(lambda m: init_weights(m, init))

    def forward(self, X):
        return self.features(X)
