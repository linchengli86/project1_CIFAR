"""Leave-one-out on the ViT recipe: how much each component is worth at 80.9%.

The CNN/MLP ladders add one trick to a raw baseline. That design does not transfer to the ViT
(its raw baseline is too weak to rank anything), so the ViT is measured the other way round:
start from its best recipe and remove one component at a time.

    python scripts/vit_loo_figure.py   # -> results/figures/vit_loo.png
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cifarlab import paths

NOISE = 0.17
ARMS = [('v1-no-warmup', 'no warmup', 'step'),
        ('v2-no-augment', 'no crop + flip', 'samples'),
        ('v3-no-ls', 'no label smoothing', 'target'),
        ('v4-no-norm', 'no z-score input', 'signal')]
COLOR = {'step': '#c0392b', 'samples': '#2e8b57', 'target': '#6a5acd', 'signal': '#1f4e79'}


def load(tag):
    p = paths.METRICS / f'ladder-vit-loo-{tag}.json'
    return json.load(open(p)) if p.exists() else None


def main():
    full = load('v0-full')
    base = 100 * full['test_acc1']
    rows = []
    for tag, label, family in ARMS:
        rec = load(tag)
        if rec:
            rows.append((label, family, 100 * rec['test_acc1'] - base))

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ypos = range(len(rows))
    for y, (label, family, delta) in zip(ypos, rows):
        ax.barh(y, delta, height=0.58, color=COLOR[family], alpha=0.92)
        ax.annotate(f'{delta:+.1f}', (delta, y), xytext=(6 if delta >= 0 else -6, 0),
                    textcoords='offset points', va='center',
                    ha='left' if delta >= 0 else 'right', fontsize=8.6, fontweight='bold')
    ax.axvspan(-NOISE, NOISE, color='#b0b0b0', alpha=0.3)
    ax.axvline(0, color='#333', lw=1.0)
    ax.set_yticks(list(ypos), [r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-13, 4)
    ax.set_xlabel('test accuracy change when the component is removed (points)')
    ax.set_title(f'ViT: remove one component from the full recipe (base {base:.1f}%)',
                 fontsize=10)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    out = paths.FIGURES / 'vit_loo.png'
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
    print(f'figure -> {out}')
    for label, _, delta in rows:
        print(f'  {label:22s} {delta:+.2f}')


if __name__ == '__main__':
    main()
