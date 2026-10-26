"""The models compared in this project, in one registry.

    from cifarlab.models import build
    net = build('resnet')          # == ResNet()
    net = build('vgg-res')         # == VGG(residual=True)

The three `*-small` entries are the low-capacity references ported from the earlier
exploration (project1). They keep their original architectures and only add `num_classes`,
so they can be trained and evaluated under this project's protocol.
"""
from functools import partial

from .cnn_small import CNN
from .googlenet import GoogLeNet
from .mixer import PatchMLP
from .mlp import MLP
from .resnet import ResNet
from .vgg import VGG
from .vit import ViT
from .vit_small import ViTSmall

__all__ = ['MODELS', 'build', 'DEFAULT_ENSEMBLE']

MODELS = {
    'googlenet': GoogLeNet,                  # 分支式：多尺度融合
    'resnet': ResNet,                        # 残差：2 卷积块 + 恒等捷径
    'vgg': VGG,                              # 纯堆叠：3 卷积块 + 池化
    'vgg-res': partial(VGG, residual=True),  # 组合：VGG 的块 + 残差捷径
    'vit': ViT,                              # 注意力：弱空间先验，位于 MLP 与 CNN 之间
    'mlp': MLP,                              # 无卷积先验的对照
    'cnn-small': CNN,                        # 2 层卷积（低容量参考）
    'vit-small': ViTSmall,                   # 2 层注意力（同宽度、1/3 深度）
    'patch-mixer': PatchMLP,                 # 同参数量，用 MLP 做 token 混合
}

DEFAULT_ENSEMBLE = ['googlenet', 'resnet', 'vgg', 'vgg-res']


def build(name, **kwargs):
    """Instantiate by name; kwargs go straight to the constructor."""
    return MODELS[name](**kwargs)
