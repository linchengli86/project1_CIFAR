"""The two-block ViT from the earlier (project1) exploration, ported unchanged.

Difference from `vit.py`: this one has no [CLS] token (it mean-pools the tokens), only two
blocks, and patchifies with `nn.Unfold` + a linear layer. Kept as a capacity reference:
same width d=128 as the main ViT, but 2 blocks instead of 7.
"""
import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_hiddens)
        self.attn = nn.MultiheadAttention(num_hiddens, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(num_hiddens)
        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens, 2 * num_hiddens),
            nn.GELU(),
            nn.Linear(2 * num_hiddens, num_hiddens),
        )

    def forward(self, X):
        Y = self.norm1(X)
        X = X + self.attn(Y, Y, Y, need_weights=False)[0]
        return X + self.mlp(self.norm2(X))


class ViTSmall(nn.Module):
    """2 blocks, d=128, 4 heads, 4x4 patches -> 64 tokens, mean-pooled."""
    def __init__(self, patch_size=4, num_hiddens=128, num_heads=4, num_layers=2,
                 num_classes=10):
        super().__init__()
        assert 32 % patch_size == 0
        num_patches = (32 // patch_size) ** 2
        self.patch = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.embedding = nn.Linear(3 * patch_size ** 2, num_hiddens)
        self.position = nn.Parameter(torch.randn(1, num_patches, num_hiddens) * 0.02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(num_hiddens, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(num_hiddens)
        self.head = nn.Linear(num_hiddens, num_classes)

    def forward_features(self, X):
        X = self.patch(X).transpose(1, 2)
        X = self.embedding(X) + self.position
        X = self.blocks(X)
        return self.norm(X).mean(dim=1)

    def forward(self, X):
        return self.head(self.forward_features(X))
