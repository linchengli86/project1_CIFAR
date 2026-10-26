"""The family figure: what each single trick is worth, for CNN / MLP / ViT.

CNN and MLP are measured by adding one trick to the raw baseline; the ViT cannot be measured
that way (its raw baseline trains poorly, so every delta would read "what it takes to revive
it"), so it is measured the other way round -- remove one component from its best recipe and
plot the value of that component. Both conventions are drawn as "how much is this trick
worth", but the baselines differ, which the panel titles state explicitly.

    python scripts/recipe_figure.py   # -> results/figures/recipe_families.png
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cifarlab import paths

NOISE = 0.17          # 100-epoch seed std, i.e. the resolution of this comparison

# (arm key, label, family) —— 顺序即图上的顺序，同族放在一起
ARMS = [
    ('norm',     'z-score input',      'signal'),
    ('bn',       'BatchNorm',          'signal'),
    ('xavier',   'Xavier init',        'init'),
    ('kaiming',  'Kaiming init',       'init'),
    ('lr',       'lr 0.05 + cosine',   'step'),
    ('adam',     'Adam 1e-3',          'step'),
    ('warmup',   'warmup (5 epochs)',  'step'),
    ('dropout',  'dropout 0.3',        'capacity'),
    ('augment',  'crop + flip',        'sample'),
    ('ls',       'label smoothing',    'target'),
]
FAMILY = {'signal': ('signal scale', '#1f4e79'), 'init': ('init scale', '#4a8fb5'),
          'step': ('step size', '#c0392b'), 'capacity': ('capacity', '#d99a2b'),
          'sample': ('samples', '#2e8b57'), 'target': ('target', '#6a5acd')}

ARMS_TAG = {
    'cnn': {'norm': 's1-norm', 'lr': 's2-lr', 'augment': 's3-augment', 'ls': 's4-ls',
            'bn': 's5-bn', 'dropout': 's6-dropout', 'xavier': 's7-xavier',
            'kaiming': 's8-kaiming'},
    'mlp': {'norm': 't1-norm', 'lr': 't2-lr', 'dropout': 't3-dropout', 'ls': 't4-ls',
            'bn': 't5-bn', 'adam': 't6-adam', 'xavier': 't7-xavier', 'kaiming': 't8-kaiming',
            'augment': 't9-augment'},
    # ViT 是 leave-one-out：去掉谁，掉多少，就是"谁值多少"
    'vit': {'warmup': 'v1-no-warmup', 'augment': 'v2-no-augment', 'ls': 'v3-no-ls',
            'norm': 'v4-no-norm'},
}

PANELS = [
    dict(ladder='cnn', kind='single', base='ladder-cnn-r0-naive', full='cnn-small',
         title='CNN (0.55M, 2 conv)'),
    dict(ladder='mlp', kind='single', base='ladder-mlp-m0-naive', full='ladder-mlp-m6-adam',
         title='MLP (0.99M)'),
    dict(ladder='vit', kind='loo', base='ladder-vit-loo-v0-full', full=None,
         title='ViT (0.94M) — value by removal'),
]


def load(tag):
    p = paths.METRICS / f'{tag}.json'
    return json.load(open(p)) if p.exists() else None


def arm_values(panel):
    """[(label, family, value)]，统一符号：正数 = 这一招有用，负数 = 有害。"""
    base = load(panel['base'])
    if base is None:
        return [], None
    base_acc = 100 * base['test_acc1']
    rows = []
    for key, label, family in ARMS:
        suffix = ARMS_TAG[panel['ladder']].get(key)
        if suffix is None:
            continue
        prefix = 'ladder-vit-loo' if panel['kind'] == 'loo' else f"ladder-{panel['ladder']}-single"
        record = load(f'{prefix}-{suffix}')
        if record is None:
            continue
        acc = 100 * record['test_acc1']
        # 加法设计：值 = 加上去之后的提升；减法设计：值 = 拿掉之后的损失
        rows.append((label, family, acc - base_acc if panel['kind'] == 'single' else base_acc - acc))
    return rows, base_acc


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.4))
    for ax, panel in zip(axes, PANELS):
        rows, base_acc = arm_values(panel)
        if not rows:
            ax.set_title(panel['title'] + '\n(runs still missing)', fontsize=9)
            continue

        ypos = np.arange(len(rows))[::-1]
        ax.axvspan(-NOISE, NOISE, color='#b0b0b0', alpha=0.30, zorder=1)
        ax.axvline(0, color='#333', lw=1.0, zorder=2)
        for y, (label, family, delta) in zip(ypos, rows):
            color = FAMILY[family][1] if delta >= 0 else '#8c3b3b'
            ax.barh(y, delta, height=0.62, color=color, alpha=0.92, zorder=3)
            ax.annotate(f'{delta:+.1f}', (delta, y), xytext=(6 if delta >= 0 else -6, 0),
                        textcoords='offset points', va='center',
                        ha='left' if delta >= 0 else 'right', fontsize=8.2,
                        color='#333', fontweight='bold')

        deltas = [d for _, _, d in rows]
        span = max(max(abs(d) for d in deltas), 1.0)
        lo, hi = min(min(deltas) - 1.8, -1.0), max(max(deltas), 0)
        summed, total = None, None
        if panel['kind'] == 'single':
            summed = sum(deltas)
            full = load(panel['full']) if panel['full'] else None
            total = 100 * full['test_acc1'] - base_acc if full is not None else None
            # 参考线只在"不把条形压扁"的范围内画，超出的用边缘标注代替
            for value in (summed, total):
                if value is None:
                    continue
                if lo - 1.8 <= value <= hi + 2.5 * span:
                    hi, lo = max(hi, value), min(lo, value)
        lo, hi = lo - 0.4, hi + 3.2
        ax.set_xlim(lo, hi)

        if panel['kind'] == 'single':
            if lo <= summed <= hi:
                ax.axvline(summed, color='#777', lw=1.2, ls=':', zorder=4)
                ax.annotate(f'sum {summed:+.1f}', (summed, len(rows) - 0.45), xytext=(4, 0),
                            textcoords='offset points', fontsize=8, color='#777', va='center')
            else:
                ax.annotate(f'sum {summed:+.1f} (off scale)', (lo + 0.2, len(rows) - 0.55),
                            fontsize=7.6, color='#777', va='center', ha='left')
            if total is not None:
                if lo <= total <= hi:
                    ax.axvline(total, color='#111', lw=1.6, ls='--', zorder=4)
                    ax.annotate(f'full recipe {total:+.1f}', (total, -0.6), xytext=(4, 0),
                                textcoords='offset points', fontsize=8.5, color='#111',
                                fontweight='bold', va='center')
                else:
                    ax.annotate(f'full recipe {total:+.1f} (off scale)',
                                (hi - 0.2, -0.6), fontsize=8, color='#111',
                                fontweight='bold', va='center', ha='right')

        ax.set_yticks(ypos, [r[0] for r in rows], fontsize=8.6)
        ax.set_ylim(-1.05, len(rows) - 0.1)
        note = (f'raw = {base_acc:.1f}%' if panel['kind'] == 'single'
                else f'full recipe = {base_acc:.1f}%')
        ax.set_title(f"{panel['title']}\n{note}", fontsize=9.5)
        ax.set_xlabel('test accuracy change (points)')
        ax.grid(axis='x', alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.92)
               for _, (_, c) in FAMILY.items()]
    labels = [name for (name, _), _ in zip(FAMILY.values(), FAMILY)]
    fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=6,
               loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('what each trick is worth  ·  CNN / MLP: added to the raw baseline  ·  '
                 f'ViT: removed from its best recipe  ·  shaded band = ±{NOISE}', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = paths.FIGURES / 'recipe_families.png'
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
    print(f'figure -> {out}')
    for panel in PANELS:
        rows, _ = arm_values(panel)
        print(f"[{panel['ladder']}]")
        for label, _, delta in rows:
            print(f'  {label:22s} {delta:+6.2f}')


if __name__ == '__main__':
    main()
