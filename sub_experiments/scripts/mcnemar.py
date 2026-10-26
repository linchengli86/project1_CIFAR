"""McNemar test between models on the same 10k test images.

Two models with close accuracies can disagree on a lot of images; the paired test asks
whether the *pattern* of disagreement is asymmetric (b vs c), which is the right question
when both are evaluated on the same samples.
"""
import math

import torch

import cifarlab as cl
from cifarlab import paths
from cifarlab.models import build

MODELS = {
    'resnet': ('resnet', 'resnet_best.pth'),
    'googlenet': ('googlenet', 'googlenet_best.pth'),
    'vgg': ('vgg', 'vgg_best.pth'),
    'vgg-res': ('vgg-res', 'vgg-res_best.pth'),
    'vit': ('vit', 'vit-adamw-w5-e150_best.pth'),
}
PAIRS = [('resnet', 'googlenet'), ('resnet', 'vgg'), ('googlenet', 'vgg'),
         ('vgg', 'vgg-res'), ('googlenet', 'vit')]

device = cl.try_gpu()
test_iter = cl.load_test_cifar10(paths.DATA)
preds = {}
for name, (model_name, ckpt) in MODELS.items():
    net = build(model_name).to(device)
    net.load_state_dict(torch.load(paths.CKPT / ckpt, map_location=device, weights_only=True))
    net.eval()
    out = []
    with torch.no_grad():
        for X, _ in test_iter:
            out.append(net(X.to(device)).argmax(1).cpu())
    preds[name] = torch.cat(out)
    print(f'{name}: loaded')

ys = torch.cat([y for _, y in test_iter])
right = {k: (v == ys) for k, v in preds.items()}

print(f'\n{"pair":28s} {"acc A":>7} {"acc B":>7} {"b":>5} {"c":>5} {"p":>8}  verdict')
for a, b in PAIRS:
    ra, rb = right[a], right[b]
    n_b = int((ra & ~rb).sum())      # A right, B wrong
    n_c = int((~ra & rb).sum())      # A wrong, B right
    if n_b + n_c == 0:
        p = 1.0
    else:
        chi2 = (abs(n_b - n_c) - 1) ** 2 / (n_b + n_c)
        p = math.erfc(math.sqrt(chi2 / 2))          # chi2 survival, df=1
    verdict = 'significant' if p < 0.01 else ('marginal' if p < 0.05 else 'not significant')
    print(f'{a + " vs " + b:28s} {ra.float().mean():7.4f} {rb.float().mean():7.4f} '
          f'{n_b:5d} {n_c:5d} {p:8.2g}  {verdict}')
