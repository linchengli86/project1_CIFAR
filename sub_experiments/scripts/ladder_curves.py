"""Did each arm actually converge? Validation curves for every recipe arm.

Reads the per-epoch tables from results/logs/ (the runs already happened), so this is a
diagnostic, not a new experiment. Each panel's side table reports the validation drift over
the last 20 epochs (mean of the last 10 minus the mean of the previous 10) -- if that is below
the noise level, the arm plateaued.

Three panels: CNN and MLP are the "add one trick to the raw baseline" arms (100 epochs);
the ViT is the leave-one-out family (150 epochs, AdamW), which is why its x-axis is longer.

    python scripts/ladder_curves.py   # -> results/figures/ladder_curves.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cifarlab import paths

ARMS = {
    'cnn': [('r0-naive', 'ladder-cnn-r0.txt', '#111111', 'naive baseline'),
            ('s1-norm', 'ladder-cnn-s1.txt', '#1f4e79', 'z-score'),
            ('s2-lr', 'ladder-cnn-s2.txt', '#c0392b', 'lr+cosine'),
            ('s3-augment', 'ladder-cnn-s3.txt', '#2e8b57', 'crop+flip'),
            ('s4-ls', 'ladder-cnn-s4.txt', '#6a5acd', 'label smoothing'),
            ('s5-bn', 'ladder-cnn-s5.txt', '#4a8fb5', 'BatchNorm'),
            ('s6-dropout', 'ladder-cnn-s6.txt', '#d99a2b', 'dropout'),
            ('s7-xavier', 'ladder-cnn-s7.txt', '#8c6d31', 'Xavier'),
            ('s8-kaiming', 'ladder-cnn-s8.txt', '#7f7f7f', 'Kaiming')],
    'mlp': [('m0-naive', 'ladder-mlp-m0.txt', '#111111', 'naive baseline'),
            ('t1-norm', 'ladder-mlp-t1.txt', '#1f4e79', 'z-score'),
            ('t2-lr', 'ladder-mlp-t2.txt', '#c0392b', 'lr+cosine'),
            ('t3-dropout', 'ladder-mlp-t3.txt', '#d99a2b', 'dropout'),
            ('t4-ls', 'ladder-mlp-t4.txt', '#6a5acd', 'label smoothing'),
            ('t5-bn', 'ladder-mlp-t5.txt', '#4a8fb5', 'BatchNorm'),
            ('t6-adam', 'ladder-mlp-t6.txt', '#e377c2', 'Adam'),
            ('t9-augment', 'ladder-mlp-t9.txt', '#2e8b57', 'crop+flip')],
    # ViT：leave-one-out 家族，150 轮、AdamW
    'vit': [('v0-full', 'loo-vit-v0.txt', '#111111', 'full recipe'),
            ('v1-no-warmup', 'loo-vit-v1.txt', '#c0392b', '- warmup'),
            ('v2-no-augment', 'loo-vit-v2.txt', '#2e8b57', '- crop+flip'),
            ('v3-no-ls', 'loo-vit-v3.txt', '#6a5acd', '- label smoothing'),
            ('v4-no-norm', 'loo-vit-v4.txt', '#1f4e79', '- z-score')],
}
TITLES = {'cnn': 'CNN arms (100 ep, added to raw)', 'mlp': 'MLP arms (100 ep, added to raw)',
          'vit': 'ViT arms (150 ep, removed from full recipe)'}


def read_log(name):
    raw = (paths.LOGS / name).read_bytes()
    text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8')
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 6 and p[0].isdigit():
            rows.append((int(p[0]), float(p[4]) * 100))
    return rows


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.0), sharey=True)
    for ax, (arch, arms) in zip(axes, ARMS.items()):
        rows = []
        for key, log, color, label in arms:
            path = paths.LOGS / log
            if not path.exists():
                continue
            curve = read_log(log)
            ep, acc = zip(*curve)
            ax.plot(ep, acc, color=color, lw=1.5, label=label, alpha=0.92)
            # 收敛判据：最后 10 轮的均值减前 10 轮的均值（比单点差更抗噪）
            tail, prev = acc[-10:], acc[-20:-10]
            drift = sum(tail) / len(tail) - sum(prev) / len(prev) if len(prev) == 10 else 0.0
            rows.append((label, acc[-1], drift))

        def flag(d):
            """Mark arms whose tail has not flattened: their delta is a moving target."""
            return ' <-- still rising' if d > 1.0 else (' <-- still falling' if d < -1.0 else '')

        # 侧表兼作图例：每行用该臂的颜色，省掉压住曲线的图例框
        palette = {label: color for _, _, color, label in arms}
        for k, (lab, end, d) in enumerate(sorted(rows, key=lambda r: -r[1])):
            ax.text(1.02, 0.74 - 0.057 * k,
                    f'{lab:17s}{end:5.1f}%  drift {d:+.1f}{flag(d)}',
                    transform=ax.transAxes, fontsize=7.2, va='center',
                    family='monospace', color=palette[lab])

        ax.set_xlabel('epoch')
        ax.set_ylabel('validation accuracy (%)')
        ax.set_ylim(20, 86)          # 三栏同一纵轴，架构之间可直接比高低
        ax.set_title(TITLES[arch], fontsize=9.5)
        ax.grid(alpha=0.25)
    fig.suptitle('every arm over its full run — flat tail = converged  ·  '
                 'drift = mean(last 10 ep) − mean(previous 10 ep)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 0.90, 0.94))
    out = paths.FIGURES / 'ladder_curves.png'
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
    print(f'figure -> {out}')
    for arch, arms in ARMS.items():
        print(f'[{arch}] {len([1 for _, lg, _, _ in arms if (paths.LOGS / lg).exists()])} arms')


if __name__ == '__main__':
    main()
