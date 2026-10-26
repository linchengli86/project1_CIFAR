"""Building blocks shared by the model definitions (d2l style)."""
from torch import nn

__all__ = ['conv_bn', 'scale', 'init_weights']


def conv_bn(cin, cout, k, stride=1, bn=True):
    """Conv(-BN)-ReLU, the unit every model below is built from.

    `k` may be a tuple, e.g. (1, 5) or (5, 1) for the factorised convolutions of
    Inception-v3; padding is always k // 2 per axis, i.e. "same" for stride 1.
    """
    kh, kw = k if isinstance(k, tuple) else (k, k)
    return [nn.Conv2d(cin, cout, (kh, kw), stride, (kh // 2, kw // 2), bias=not bn),
            nn.BatchNorm2d(cout) if bn else nn.Identity(), nn.ReLU(inplace=True)]


def scale(c, width):
    """Scale a channel count (or a tuple of them) by `width`, rounded to a multiple of 8."""
    return tuple(scale(x, width) for x in c) if isinstance(c, tuple) else max(8, int(round(c * width / 8)) * 8)


def init_weights(m, kind):
    """Weight initialisation selected by the `init` argument ('xavier' | 'kaiming')."""
    if not isinstance(m, (nn.Conv2d, nn.Linear)):
        return
    if kind == 'xavier':
        nn.init.xavier_uniform_(m.weight)
    elif kind == 'kaiming':
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
        nn.init.zeros_(m.bias)
