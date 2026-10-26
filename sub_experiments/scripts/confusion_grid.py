"""Row-normalised confusion matrices of the four models, on CIFAR-10 and CIFAR-100.

    python confusion_grid.py     # -> results/figures/confusion_cifar10.png
                                 #    results/figures/confusion_cifar100.png
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


def cifar100_names():
    with open(paths.DATA100 / 'meta', 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    return [n.decode() for n in meta[b'fine_label_names']]

RUNS = {
    'cifar10': [('resnet', 'ResNet', 'resnet_best.pth'),
                ('googlenet', 'GoogLeNet', 'googlenet_best.pth'),
                ('vgg', 'VGG', 'vgg_best.pth'),
                ('vit', 'ViT', 'vit-adamw-w5-e150_best.pth')],
    'cifar100': [('resnet', 'ResNet', 'resnet-c100_best.pth'),
                 ('googlenet', 'GoogLeNet', 'googlenet-c100_best.pth'),
                 ('vgg', 'VGG', 'vgg-c100_best.pth'),
                 ('vit', 'ViT', 'vit-adamw-w5-e150-c100_best.pth')],
}


def confusion(net, loader, device, n):
    net.eval()
    cm = np.zeros((n, n), dtype=np.int64)
    with torch.no_grad():
        for X, y in loader:
            pred = net(X.to(device)).argmax(1).cpu().numpy()
            np.add.at(cm, (y.numpy(), pred), 1)
    return cm


def main():
    device = cl.try_gpu()
    global NAMES
    NAMES = cifar100_names()
    for dataset, n in (('cifar10', 10), ('cifar100', 100)):
        root = paths.DATA if dataset == 'cifar10' else paths.DATA100
        loader = cl.load_test_cifar10(root, batch_size=512, dataset=dataset)
        fig, axes = plt.subplots(2, 2, figsize=(11 if n == 10 else 11.6, 10.2 if n == 10 else 10.8))
        for ax, (key, label, ckpt) in zip(axes.flat, RUNS[dataset]):
            net = build(key, num_classes=n).to(device)
            net.load_state_dict(torch.load(paths.CKPT / ckpt, map_location=device,
                                           weights_only=True))
            cm = confusion(net, loader, device, n)
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1.0 if n == 10 else 0.8)
            if n == 10:
                for i in range(n):
                    for j in range(n):
                        if norm[i, j] > 0.01:
                            ax.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center',
                                    fontsize=6.2,
                                    color='white' if norm[i, j] > 0.5 else '#333')
                ax.set_xticks(range(n), CLASSES, rotation=45, ha='right', fontsize=7.5)
                ax.set_yticks(range(n), CLASSES, fontsize=7.5)
            else:
                ax.set_xticks(range(0, n, 10), range(0, n, 10), fontsize=6.5)
                ax.set_yticks(range(0, n, 10), range(0, n, 10), fontsize=6.5)
                top = np.argsort(norm, axis=1)[:, -2]                 # 每个类最强的错分
                worst = np.argsort(-norm[np.arange(n), top])[:3]
                ax.set_title(f'{label}   top-1 {np.trace(cm) / cm.sum():.3f}',
                             fontsize=9.5, color=COLOR[key], fontweight='bold')
                ax.text(0.0, -0.13, 'worst: ' + ', '.join(
                    f'{NAMES[i]}→{NAMES[top[i]]} {norm[i, top[i]]:.2f}' for i in worst),
                    transform=ax.transAxes, fontsize=6.5, color='#666')
            if n == 10:
                acc = np.trace(cm) / cm.sum()
                ax.set_title(f'{label}   top-1 {acc:.3f}', fontsize=9.5, color=COLOR[key],
                             fontweight='bold')
            ax.tick_params(length=2)
        fig.supxlabel('predicted', fontsize=9)
        fig.supylabel('true', fontsize=9)
        fig.subplots_adjust(hspace=0.34 if n == 10 else 0.30, wspace=0.16)
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label='fraction of true class')
        fig.suptitle(f'Confusion matrices on {"CIFAR-10" if n == 10 else "CIFAR-100"} test set '
                     f'(row-normalised)', fontsize=11, y=0.95)
        out = paths.FIGURES / f'confusion_{dataset}.png'
        fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'{dataset}: figure -> {out}')


if __name__ == '__main__':
    main()
