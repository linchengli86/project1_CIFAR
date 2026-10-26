"""Non-convolutional baseline, used for the inductive-bias comparison."""
from torch import nn

from .layers import init_weights


class MLP(nn.Module):
    """3*32*32 -> hidden -> 10 on raw pixels: same capacity, no translation prior.

    With hidden=320 it has 0.99M parameters, i.e. the same capacity as GoogLeNet (1.01M),
    so the accuracy gap on CIFAR-10 measures the value of the conv / weight-sharing bias
    rather than a difference in size. `bn=True` adds BatchNorm after the hidden layer,
    used by the recipe ladder to isolate that trick.
    """
    def __init__(self, hidden=320, num_classes=10, dropout=0.0, bn=False, init=None):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(3 * 32 * 32, hidden)]
        if bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)
        if init:
            self.apply(lambda m: init_weights(m, init))

    def forward(self, x):
        return self.net(x)
