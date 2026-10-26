"""在 CIFAR-10 测试集上评估一个 checkpoint。

    python eval.py                      # 默认 googlenet
    python eval.py resnet               # 其余模型：resnet / vgg / vgg-res

输出 top-1 / top-3、每类 precision / recall / F1、macro 平均，并把行归一化的混淆矩阵存到
results/figures/<checkpoint 名>_confusion.png。
"""
import sys
from pathlib import Path

import numpy as np
import torch

import cifarlab as cl
from cifarlab import paths
from cifarlab.models import build

CLASSES = 'airplane automobile bird cat deer dog frog horse ship truck'.split()
NUM = 10


def confusion(net, data_iter, device, num=10):
    """(行 = 真实类, 列 = 预测类) 的混淆矩阵，以及 top-3 命中数。"""
    net.eval()
    cm, top3 = np.zeros((num, num), dtype=np.int64), 0
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X.to(device)).cpu()
            top3 += (y_hat.topk(3, 1).indices == y[:, None]).any(1).sum().item()
            np.add.at(cm, (y.numpy(), y_hat.argmax(1).numpy()), 1)
    return cm, top3


def plot(cm, path, classes=None, model=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(cm)
    classes = classes or [str(i) for i in range(n)]
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4) if n <= 10 else (9, 8))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    if n <= 10:
        for i in range(n):
            for j in range(n):
                if norm[i, j] > 0.01:
                    ax.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center', fontsize=6,
                            color='white' if norm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(n), classes, rotation=45 if n <= 10 else 90, ha='right', fontsize=8)
    ax.set_yticks(range(n), classes, fontsize=8)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    name = f'{model}  ·  ' if model else ''      # 标题必须带模型名，否则单张图无法辨识
    ax.set_title(f'{name}test top-1 {np.trace(cm) / cm.sum():.3f}  ({n} classes)')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'confusion matrix -> {path}')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = dict(a[2:].split('=') for a in sys.argv[1:] if a.startswith('--'))
    name = args[0] if args else 'googlenet'
    dataset = flags.get('dataset', 'cifar10')
    n_classes = 10 if dataset == 'cifar10' else 100
    root = paths.DATA if dataset == 'cifar10' else paths.DATA100
    path = Path(args[1]) if len(args) > 1 else paths.CKPT / f'{name}_best.pth'
    device = cl.try_gpu()
    net = build(name, num_classes=n_classes).to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    test_iter = cl.load_test_cifar10(root, dataset=dataset)
    cm, top3 = confusion(net, test_iter, device, n_classes)
    tp, n = np.diag(cm), cm.sum()
    prec, rec = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    print(f'{"class":11s} {"precision":>9} {"recall":>7} {"f1":>6} {"n":>5}')
    for i, c in enumerate(CLASSES if n_classes == 10 else range(n_classes)):
        print(f'{c:11s} {prec[i]:9.3f} {rec[i]:7.3f} {f1[i]:6.3f} {cm[i].sum():5d}')
    print(f'{"macro":11s} {prec.mean():9.3f} {rec.mean():7.3f} {f1.mean():6.3f}')
    print(f'top-1 {tp.sum() / n:.4f}   top-3 {top3 / n:.4f}   ({n} test images)')
    plot(cm, paths.FIGURES / f'{path.stem}_confusion.png',
         CLASSES if n_classes == 10 else None, model=name)


if __name__ == '__main__':
    main()
