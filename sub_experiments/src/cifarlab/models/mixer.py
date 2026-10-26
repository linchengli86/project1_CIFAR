"""Patch-MLP (MLP-Mixer style) from the earlier exploration, ported unchanged.

Same token embedding as `vit_small`, but attention is replaced by a token-mixing MLP:
it mixes information across the 64 patch positions with a learned 64->512->64 map.
A useful contrast: identical parameter count (~0.28M) and depth, different mixing operator.
"""
from torch import nn

from .vit_small import ViTSmall


class MixerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(128)
        self.token_mlp = nn.Sequential(
            nn.Linear(64, 512), nn.GELU(), nn.Linear(512, 64),
        )
        self.norm2 = nn.LayerNorm(128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128),
        )

    def forward(self, X):
        Y = self.norm1(X).transpose(1, 2)
        X = X + self.token_mlp(Y).transpose(1, 2)
        return X + self.mlp(self.norm2(X))


class PatchMLP(ViTSmall):
    """`vit_small` with its two attention blocks replaced by mixer blocks."""
    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes)
        self.blocks = nn.Sequential(MixerBlock(), MixerBlock())
