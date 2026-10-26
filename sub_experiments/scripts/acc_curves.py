"""All eight runs in one axes: four models x (CIFAR-10 solid, CIFAR-100 dashed).

    python acc_curves.py     # -> results/figures/acc_curves.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cifarlab import paths

COLOR = {'resnet': '#1f4e79', 'googlenet': '#c0392b', 'vgg': '#d99a2b', 'vit': '#6a5acd'}
LABEL = {'resnet': 'ResNet', 'googlenet': 'GoogLeNet', 'vgg': 'VGG', 'vit': 'ViT'}
LOG = {
    'cifar10': {'resnet': 'resnet', 'googlenet': 'googlenet', 'vgg': 'vgg',
                'vit': 'vit-adamw-w5-e150'},
    'cifar100': {'resnet': 'resnet-c100', 'googlenet': 'googlenet-c100', 'vgg': 'vgg-c100',
                 'vit': 'vit-c100'},
}


def read_log(tag):
    raw = (paths.LOGS / f'{tag}.txt').read_bytes()
    text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8')
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 6 and p[0].isdigit():
            rows.append((int(p[0]), float(p[4]) * 100))          # ep, val
    return rows


fig, ax = plt.subplots(figsize=(10.2, 5.2))
for dataset, ls, alpha in (('cifar10', '-', 1.0), ('cifar100', '--', 0.85)):
    for key in ('resnet', 'googlenet', 'vgg', 'vit'):
        ep, va = zip(*read_log(LOG[dataset][key]))
        ax.plot(ep, va, color=COLOR[key], ls=ls, lw=1.7 if ls == '-' else 1.5,
                alpha=alpha, label=f'{LABEL[key]} · {"CIFAR-10" if ls == "-" else "CIFAR-100"}')
    ax.annotate('CIFAR-10' if ls == '-' else 'CIFAR-100',
                (152, [v for _, v in read_log(LOG[dataset]['resnet'])][-1]),
                fontsize=8.5, color='#8a94a0', va='center')

ax.set_xlabel('epoch')
ax.set_ylabel('validation accuracy (%)')
ax.set_title('validation accuracy vs epoch · four models on two datasets', fontsize=10.5)
ax.set_ylim(25, 97)
ax.set_xlim(0, 168)
ax.grid(alpha=0.22)

models = [Line2D([], [], color=COLOR[k], lw=1.8, label=LABEL[k]) for k in LABEL]
styles = [Line2D([], [], color='#555', lw=1.8, ls='-', label='CIFAR-10 (5000 / class)'),
          Line2D([], [], color='#555', lw=1.5, ls='--', label='CIFAR-100 (500 / class)')]
fig.legend(handles=models + styles, frameon=False, fontsize=9, ncol=6,
           loc='lower center', bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = paths.FIGURES / 'acc_curves.png'
fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
print(f'figure -> {out}')
for dataset in ('cifar10', 'cifar100'):
    row = [f'{LABEL[k]} {max(v for _, v in read_log(LOG[dataset][k])):.2f}'
           for k in ('resnet', 'googlenet', 'vgg', 'vit')]
    print(f'  {dataset:9s} best val: ' + ' | '.join(row))
