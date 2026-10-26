"""The three basic models on one axes: train vs validation, end to end.

Reads the per-epoch logs (the runs already happened). The point is the *gap* between the two
curves: MLP never fits the training set (underfitting), CNN and ViT reach ~100% train while
validation stalls 14-20 points lower (memorisation). Same plot, same axes, so the three are
directly comparable.

    python scripts/basic_curves.py   # -> results/figures/basic_models.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cifarlab import paths

# (图例名, 日志, 颜色, 备注)
RUNS = [
    ('MLP', 'ladder-mlp-m6.txt', '#7f7f7f'),
    ('CNN', 'cnn-small.txt', '#1f4e79'),
    ('ViT', 'vit-adamw-w5-e150.txt', '#6a5acd'),
]


def read_log(name):
    raw = (paths.LOGS / name).read_bytes()
    text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8')
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 6 and p[0].isdigit():
            rows.append((int(p[0]), float(p[3]) * 100, float(p[4]) * 100))   # epoch, train, val
    return rows


def main():
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for label, log, color in RUNS:
        path = paths.LOGS / log
        if not path.exists():
            continue
        curve = read_log(log)
        ep = [r[0] for r in curve]
        tr = [r[1] for r in curve]
        va = [r[2] for r in curve]
        ax.plot(ep, tr, color=color, lw=1.9, label=f'{label} · train')
        ax.plot(ep, va, color=color, lw=1.9, ls='--', label=f'{label} · val')
        ax.annotate(f'{va[-1]:.1f}', (ep[-1], va[-1]), xytext=(5, -2),
                    textcoords='offset points', fontsize=8, color=color, fontweight='bold')

    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy (%)')
    ax.set_xlim(0, 158)
    ax.set_ylim(0, 104)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc='lower right')
    ax.set_title('MLP / CNN / ViT: solid = train, dashed = validation', fontsize=10)
    fig.tight_layout()
    out = paths.FIGURES / 'basic_models.png'
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
    print(f'figure -> {out}')
    for label, log, _ in RUNS:
        c = read_log(log)
        print(f'  {label:4s} final train {c[-1][1]:5.1f}%  val {c[-1][2]:5.1f}%  '
              f'gap {c[-1][1] - c[-1][2]:+5.1f}')


if __name__ == '__main__':
    main()
