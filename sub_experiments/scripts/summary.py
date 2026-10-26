"""把 results/metrics/*.json 汇总成报告表格和一张柱状图。

    python summary.py            # 表格打到 stdout，柱状图存 results/figures/study_summary.png
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cifarlab import paths

GROUPS = [('optimizer', ['base', 'opt-adam', 'opt-adamw', 'opt-rmsprop']),
          ('lr / weight decay', ['base', 'lr-0.01', 'lr-0.1', 'wd-0', 'wd-5e-3']),
          ('capacity / inductive bias', ['width-0.5', 'base', 'width-1.5', 'mlp', 'mlp-adam']),
          ('norm / dropout / init', ['base', 'no-bn', 'dropout-0', 'dropout-0.5',
                                     'init-xavier', 'init-kaiming'])]


def main():
    rows = {}
    for path in paths.METRICS.glob('*.json'):
        with open(path) as f:
            j = json.load(f)
        if 'name' in j:
            rows[j['name']] = j
    for title, names in GROUPS:
        print(f'\n### {title}')
        print('| config | params (M) | best val acc | last val acc | acc@3 | epochs |')
        print('| --- | ---: | ---: | ---: | ---: | ---: |')
        for n in names:
            j = rows.get(n)
            if not j:
                print(f'| {n} | - | - | - | - | - |')
                continue
            print(f'| {n} | {j["params"] / 1e6:.2f} | {j["val_acc"]:.4f} | '
                  f'{j["val_acc_last"]:.4f} | {j["val_acc3_last"]:.4f} | {j["epochs"]} |')

    names = sorted(rows, key=lambda n: rows[n]['val_acc'])
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(names) + 1.2))
    ax.barh(names, [rows[n]['val_acc'] * 100 for n in names],
            color=['tab:orange' if n == 'base' else 'tab:blue' for n in names])
    for i, n in enumerate(names):
        ax.text(rows[n]['val_acc'] * 100 + 0.3, i, f'{rows[n]["val_acc"] * 100:.1f}',
                va='center', fontsize=7)
    ax.set_xlabel('validation accuracy (%)')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.FIGURES / 'study_summary.png', dpi=150, bbox_inches='tight')
    print(f'\nchart -> {paths.FIGURES / "study_summary.png"}')


if __name__ == '__main__':
    main()
