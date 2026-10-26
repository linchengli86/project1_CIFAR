"""Where do a model's errors land? Semantic structure of the error mass on CIFAR-100.

"Retention" of a superclass = among all errors on that superclass's 5 fine classes, the share
that lands on another class of the SAME superclass.  4/99 = 4% is the random baseline.

    python semantic_errors.py   # -> results/figures/semantic_errors.png
"""
import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import cifarlab as cl
from cifarlab import paths
from cifarlab.models import build

RUNS = [('resnet', 'ResNet', 'resnet-c100_best.pth'),
        ('googlenet', 'GoogLeNet', 'googlenet-c100_best.pth'),
        ('vgg', 'VGG', 'vgg-c100_best.pth'),
        ('vit', 'ViT', 'vit-adamw-w5-e150-c100_best.pth')]
COLOR = {'resnet': '#1f4e79', 'googlenet': '#c0392b', 'vgg': '#d99a2b', 'vit': '#6a5acd'}


def superclass_map():
    with open(paths.DATA100 / 'meta', 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    names = [n.decode() for n in meta[b'coarse_label_names']]
    with open(paths.DATA100 / 'train', 'rb') as f:
        tr = pickle.load(f, encoding='bytes')
    cl, fl = np.asarray(tr[b'coarse_labels']), np.asarray(tr[b'fine_labels'])
    f2c = np.zeros(100, dtype=int)
    for c in range(100):
        v, n = np.unique(cl[fl == c], return_counts=True)
        f2c[c] = v[n.argmax()]
    return names, f2c


def confusion(net, loader, device):
    net.eval()
    cm = np.zeros((100, 100), dtype=np.int64)
    with torch.no_grad():
        for X, y in loader:
            p = net(X.to(device)).argmax(1).cpu().numpy()
            np.add.at(cm, (y.numpy(), p), 1)
    return cm.astype(float)


def main():
    names, f2c = superclass_map()
    device = cl.try_gpu()
    loader = cl.load_test_cifar10(paths.DATA100, batch_size=512, dataset='cifar100')

    with open(paths.DATA100 / 'meta', 'rb') as f:
        fine = [n.decode() for n in pickle.load(f, encoding='bytes')[b'fine_label_names']]

    ret, cms = {}, {}
    for key, label, ckpt in RUNS:
        net = build(key, num_classes=100).to(device)
        net.load_state_dict(torch.load(paths.CKPT / ckpt, map_location=device, weights_only=True))
        cm = confusion(net, loader, device)
        cms[key] = cm
        prof = np.zeros(20)
        for c in range(20):
            rows = np.where(f2c == c)[0]
            sub = cm[np.ix_(rows, rows)]
            errors = cm[rows].sum() - np.trace(sub)
            prof[c] = (sub.sum() - np.trace(sub)) / errors if errors > 0 else np.nan
        ret[key] = prof
        print(f'{label:10s} overall retention {prof.mean():.3f}   (chance 0.040)')

    PAIRS = [('man', 'boy'), ('woman', 'girl'), ('girl', 'boy'), ('boy', 'girl'),
             ('maple_tree', 'oak_tree'), ('tulip', 'rose'), ('streetcar', 'bus')]
    pairs = []
    for a, b in PAIRS:
        i, j = fine.index(a), fine.index(b)
        raw, cond = {}, {}
        for key, _, _ in RUNS:
            cm = cms[key]
            raw[key] = round(100 * cm[i, j] / cm[i].sum(), 1)
            cond[key] = round(100 * cm[i, j] / (cm[i].sum() - cm[i, i]), 1)
        pairs.append({'from': a, 'to': b, 'raw': raw, 'cond': cond})

    dump = {'superclasses': names, 'retention': {k: [round(x, 4) for x in v] for k, v in ret.items()},
            'overall': {k: round(float(v.mean()), 4) for k, v in ret.items()},
            'pairs': pairs}
    out_json = paths.METRICS / 'semantic_errors.json'
    with open(out_json, 'w') as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    print(f'data -> {out_json}')

    order = np.argsort(-ret['resnet'])
    mat = np.array([ret[k][order] for k, _, _ in RUNS]).T          # 20 x 4

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), gridspec_kw={'width_ratios': [1.05, 1]})

    ax = axes[0]
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=0.85, aspect='auto')
    for i in range(20):
        for j in range(4):
            v = mat[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.6,
                    color='white' if v > 0.45 else '#333')
    ax.set_xticks(range(4), [l for _, l, _ in RUNS], fontsize=8.5, rotation=20)
    ax.set_yticks(range(20), [names[c][:26] for c in order], fontsize=7.8)
    ax.set_title('(a) share of a superclass\'s errors that stay inside it', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    ax = axes[1]
    vit = ret['vit'][order]
    cnn = np.mean([ret[k][order] for k in ('resnet', 'googlenet', 'vgg')], axis=0)
    diff = vit - cnn
    ypos = np.arange(20)[::-1]
    ax.barh(ypos, diff, color=['#6a5acd' if d > 0 else '#c8c8c8' for d in diff], height=0.62)
    ax.axvline(0, color='#333', lw=0.9)
    ax.set_yticks(ypos, [names[c][:26] for c in order], fontsize=7.8)
    ax.set_xlabel('ViT retention − mean CNN retention')
    ax.set_title('(b) where ViT keeps the error inside the neighbourhood', fontsize=10)
    ax.grid(axis='x', alpha=0.25)

    fig.tight_layout()
    out = paths.FIGURES / 'semantic_errors.png'
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
    print(f'figure -> {out}')
    print('\nViT 保留得更多的超类:',
          ', '.join(names[c] for c in np.array(order)[diff > 0]))


if __name__ == '__main__':
    main()
