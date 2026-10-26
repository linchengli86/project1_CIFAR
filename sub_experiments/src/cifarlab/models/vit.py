"""A small Vision Transformer adapted to 32x32 CIFAR-10, in d2l style.

The ImageNet ViT patches 224x224 into 16x16 tiles (196 tokens) with a 768-wide, 12-block
body (86M parameters).  For 32x32 that recipe leaves 4 patches per side, so here:

  * patch = 4  ->  8x8 = 64 tokens, and a 4x4/4 convolution does the patchify,
  * d = 128    ->  the whole budget (~1M) goes into 4 heads x 6-7 blocks,
  * no big MLP head: a single linear layer on the [CLS] token, like the other models.

The point of including it is the inductive-bias axis: an MLP has no spatial prior at all,
a CNN has locality + weight sharing built in, and this ViT sits in between -- attention can
learn a spatial prior, but only from the data.
"""
import torch
from torch import nn


class PatchEmbed(nn.Module):
    """Cut the image into non-overlapping patches and project each to a token.

    A stride-`p` p-convolution is exactly patchify + linear projection, so no unfolding
    is needed: (B, 3, 32, 32) -> (B, 64, d).
    """
    def __init__(self, patch, cin, d):
        super().__init__()
        self.proj = nn.Conv2d(cin, d, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Block(nn.Module):
    """Pre-norm transformer block: attention then MLP, both with a residual path."""
    def __init__(self, d, heads, ratio=2, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * ratio), nn.GELU(), nn.Linear(d * ratio, d))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class ViT(nn.Module):
    """Patch embedding + learnable position + [CLS] token + `depth` blocks."""
    def __init__(self, num_classes=10, patch=4, d=128, depth=7, heads=4, ratio=2,
                 dropout=0.3, img_size=32):
        super().__init__()
        n_tokens = (img_size // patch) ** 2
        self.patch_embed = PatchEmbed(patch, 3, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d))
        self.blocks = nn.Sequential(*[Block(d, heads, ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout), nn.Linear(d, num_classes))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls.expand(x.shape[0], -1, -1), x), 1)
        x = self.norm(self.blocks(x + self.pos))
        return self.head(x[:, 0])
