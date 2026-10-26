"""CIFAR-10 -> CIFAR-100 comparison: slope chart, overfitting gap, and val curves."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cifarlab import paths

KEYS = ['resnet', 'googlenet', 'vgg', 'vit']
LABEL = {'resnet': 'ResNet', 'googlenet': 'GoogLeNet', 'vgg': 'VGG', 'vit': 'ViT'}
COLOR = {'resnet': '#1f4e79', 'googlenet': '#c0392b', 'vgg': '#d99a2b', 'vit': '#6a5acd'}
MARK = {'resnet': 'o', 'googlenet': 's', 'vgg': '^', 'vit': 'D'}

C10 = {'resnet': 93.65, 'googlenet': 92.44, 'vgg': 92.64, 'vit': 80.14}
STD10 = {'resnet': 0.08, 'googlenet': 0.17, 'vgg': 0.15, 'vit': 0.0}
METRIC = {'resnet': 'resnet-c100', 'googlenet': 'googlenet-c100', 'vgg': 'vgg-c100',
          'vit': 'vit-adamw-w5-e150-c100'}
LOG = {'resnet': 'resnet-c100', 'googlenet': 'googlenet-c100', 'vgg': 'vgg-c100',
       'vit': 'vit-c100'}


def read_log(tag):
    raw = (paths.LOGS / f'{tag}.txt').read_bytes()
    text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8')
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 6 and p[0].isdigit():
            rows.append((int(p[0]), float(p[3]) * 100, float(p[4]) * 100))   # ep, train, val
    return rows


C100, TR100 = {}, {}
for k in KEYS:
    C100[k] = 100 * json.load(open(paths.METRICS / f'{METRIC[k]}.json'))['test_acc1']
    TR100[k] = read_log(LOG[k])[-1][1]

fig = plt.figure(figsize=(15.2, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 1.15], wspace=0.30)

# ------------------------------------------------------------- (a) grouped bars
ax = fig.add_subplot(gs[0, 0])
order = sorted(KEYS, key=lambda k: -C10[k])
h = 0.36
for i, k in enumerate(order):
    y = len(order) - 1 - i
    ax.barh(y + h / 2, C10[k], height=h, color=COLOR[k], alpha=1.0, zorder=3)
    ax.barh(y - h / 2, C100[k], height=h, color=COLOR[k], alpha=0.45, zorder=3)
    ax.errorbar([C10[k]], [y + h / 2], xerr=[STD10[k]], fmt='none', ecolor='#333',
                elinewidth=1.2, capsize=3, zorder=5)
    ax.annotate(f'{C10[k]:.1f}', (C10[k] - 1.4, y + h / 2), va='center', ha='right',
                fontsize=8.2, color='white', fontweight='bold', zorder=6)
    ax.annotate(f'{C100[k]:.1f}', (C100[k] + 1.0, y - h / 2), va='center',
                fontsize=8.2, color=COLOR[k], fontweight='bold')
ax.set_yticks(range(len(order)), [LABEL[k] for k in order[::-1]], fontsize=9)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlim(45, 104)
ax.set_xlabel('test top-1 accuracy (%)')
ax.set_title('(a) same architecture, 10x fewer images per class', fontsize=9.5)
ax.grid(axis='x', alpha=0.25)
ax.legend(handles=[Patch(facecolor='#555', label='CIFAR-10 (5000 / class)'),
                   Patch(facecolor='#555', alpha=0.45, label='CIFAR-100 (500 / class)')],
          frameon=False, fontsize=7.8, loc='lower right')

# ------------------------------------------------------- (b) overfitting gap
ax = fig.add_subplot(gs[0, 1])
order = sorted(KEYS, key=lambda k: -C100[k])
for i, k in enumerate(order):
    y = len(order) - 1 - i
    ax.plot([C100[k], TR100[k]], [y, y], color=COLOR[k], lw=2.4, alpha=0.40, zorder=2,
            solid_capstyle='round')
    ax.plot([C100[k]], [y], 'o', color=COLOR[k], ms=8.5, zorder=4)
    ax.plot([TR100[k]], [y], 'o', mfc='white', mec=COLOR[k], mew=1.8, ms=8.5, zorder=4)
    ax.annotate(f'{C100[k]:.1f}', (C100[k], y), xytext=(0, -15), textcoords='offset points',
                ha='center', fontsize=8, color=COLOR[k], fontweight='bold')
    ax.annotate(f'gap {TR100[k] - C100[k]:.0f}', ((C100[k] + TR100[k]) / 2, y), xytext=(0, 9),
                textcoords='offset points', ha='center', fontsize=7.8, color='#555')
ax.set_yticks(range(len(order)), [LABEL[k] for k in order[::-1]], fontsize=9)
ax.set_ylim(-0.65, len(order) - 0.35)
ax.set_xlim(45, 104)
ax.set_xlabel('accuracy on CIFAR-100 (%)')
ax.set_title('(b) CIFAR-100: test vs train accuracy', fontsize=9.5)
ax.grid(axis='x', alpha=0.25)
ax.legend(handles=[Line2D([], [], marker='o', ls='none', color='#555', ms=8, label='test (top-1)'),
                   Line2D([], [], marker='o', ls='none', mfc='white', mec='#555', mew=1.8,
                          ms=8, label='train')],
          frameon=False, fontsize=7.8, loc='lower right')

# ------------------------------------------------------------------ (c) curves
ax = fig.add_subplot(gs[0, 2])
for k in KEYS:
    ep, _, va = zip(*read_log(LOG[k]))
    ax.plot(ep, va, color=COLOR[k], lw=1.8, label=LABEL[k])
ax.set_xlabel('epoch')
ax.set_ylabel('validation accuracy (%)')
ax.set_title('(c) CIFAR-100 training curves', fontsize=9.5)
ax.set_ylim(25, 78)
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8, loc='lower right')

fig.savefig(paths.FIGURES / 'cifar10_vs_cifar100.png', dpi=180, bbox_inches='tight',
            facecolor='white')
print('figure -> results/figures/cifar10_vs_cifar100.png')
for k in order:
    print(f'  {LABEL[k]:10s} C10 {C10[k]:6.2f}  C100 {C100[k]:6.2f}  '
          f'drop {C10[k] - C100[k]:5.2f}  train {TR100[k]:5.1f}')
