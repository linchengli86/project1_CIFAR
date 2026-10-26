"""The biggest confusion pairs: how often each model confuses class i for class j.

    python confusion_pairs.py    # -> results/figures/confusion_pairs.png

The heatmaps show everything; this figure extracts the pairs that actually carry the error
mass, so the four models can be compared pair by pair.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

import cifarlab as cl
from cifarlab import paths
from cifarlab.models import build

CLASSES = 'airplane automobile bird cat deer dog frog horse ship truck'.split()
COLOR = {'resnet': '#1f4e79', 'googlenet': '#c0392b', 'vgg': '#d99a2b', 'vit': '#6a5acd'}
LABEL = {'resnet': 'ResNet', 'googlenet': 'GoogLeNet', 'vgg': 'VGG', 'vit': 'ViT'}
RUNS = {
    'cifar10': [('resnet', 'resnet_best.pth'), ('googlenet', 'googlenet_best.pth'),
                ('vgg', 'vgg_best.pth'), ('vit', 'vit-adamw-w5-e150_best.pth')],
    'cifar100': [('resnet', 'resnet-c100_best.pth'), ('googlenet', 'googlenet-c100_best.pth'),
                 ('vgg', 'vgg-c100_best.pth'), ('vit', 'vit-adamw-w5-e150-c100_best.pth')],
}


def names_for(dataset):
    if dataset == 'cifar10':
        return CLASSES
    with open(paths.DATA100 / 'meta', 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    return [n.decode() for n in meta[b'fine_label_names']]


def confusion(net, loader, device, n):
    net.eval()
    cm = np.zeros((n, n), dtype=np.int64)
    with torch.no_grad():
        for X, y in loader:
            pred = net(X.to(device)).argmax(1).cpu().numpy()
            np.add.at(cm, (y.numpy(), pred), 1)
    return cm / np.maximum(cm.sum(1, keepdims=True), 1)


def main(top_k=7):
    device = cl.try_gpu()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    for ax, dataset in zip(axes, ('cifar10', 'cifar100')):
        n = 10 if dataset == 'cifar10' else 100
        names = names_for(dataset)
        root = paths.DATA if dataset == 'cifar10' else paths.DATA100
        loader = cl.load_test_cifar10(root, batch_size=512, dataset=dataset)

        mats = {}
        for key, ckpt in RUNS[dataset]:
            net = build(key, num_classes=n).to(device)
            net.load_state_dict(torch.load(paths.CKPT / ckpt, map_location=device,
                                           weights_only=True))
            mats[key] = confusion(net, loader, device, n)

        off = np.zeros((n, n))
        for m in mats.values():
            mm = m.copy()
            np.fill_diagonal(mm, 0)
            off += mm / len(mats)                          # 四模型平均，避免被 ViT 主导
        idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0]
        picked, seen = [], set()
        for i, j in idx:
            if off[i, j] < 0.03:
                break
            if names[i] in seen:                            # 每个真实类只保留最强的一条
                continue
            seen.add(names[i])
            picked.append((i, j))
            if len(picked) == top_k:
                break

        ypos = np.arange(len(picked))[::-1]
        h = 0.19
        for k_i, key in enumerate(('resnet', 'googlenet', 'vgg', 'vit')):
            vals = [mats[key][i, j] * 100 for i, j in picked]
            ax.barh(ypos + (1.5 - k_i) * h, vals, height=h, color=COLOR[key], label=LABEL[key])
        ax.set_yticks(ypos, [f'{names[i]} → {names[j]}' for i, j in picked], fontsize=8.5)
        ax.set_xlabel('share of the true class misclassified (%)')
        ax.set_title(f'({"ab"[dataset == "cifar100"]}) {"CIFAR-10" if dataset == "cifar10" else "CIFAR-100"}'
                     ' · strongest confusion pairs', fontsize=10)
        ax.grid(axis='x', alpha=0.25)
        ax.set_xlim(0, max(max(mats[k][i, j] for i, j in picked) for k in mats) * 100 * 1.12)

    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in LABEL],
               labels=list(LABEL.values()), frameon=False, fontsize=9, ncol=4,
               loc='lower center', bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    fig.tight_layout()
    out = paths.FIGURES / 'confusion_pairs.png'
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
    print(f'figure -> {out}')


if __name__ == '__main__':
    main()
