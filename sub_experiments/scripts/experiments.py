"""17 组对照实验：每组都是同一个 24-epoch 协议，只改一个变量。

    python experiments.py                 # 依次跑完 17 组
    python experiments.py base opt-adam   # 只跑这几组（可以开多个进程并行）
    python experiments.py base --seed=1   # 换种子重跑同一组，用来量 24 轮协议的噪声底

种子默认为 0（历史产物都是这个种子）；换种子时产物名带 `-seed<N>` 后缀，互不覆盖。

产物：results/metrics/<name>.json（配置 + 指标）、results/checkpoints/ablations/<name>.pth、
results/tb/<name>（TensorBoard，含 hparams 面板）。
"""
import json
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

import cifarlab as cl
from cifarlab import paths
from cifarlab.models.mlp import MLP
from cifarlab.models import build

BASE = dict(epochs=24, batch_size=128, lr=0.05, opt='sgd', wd=5e-4, ls=0.1,
            width=1.0, dropout=0.3, bn=True, init=None, model='googlenet', augment=True)

CONFIGS = {
    'base': {},                                          # 后面所有组的参照点
    'no-aug': dict(augment=False),                       # --- 数据：去掉裁剪与翻转 ---
    'resnet': dict(model='resnet'),                      # --- 架构：同预算的残差网络 ---
    'resnet-width-1.5': dict(model='resnet', width=1.5),
    'opt-adam': dict(opt='adam', lr=1e-3),               # --- 优化器 ---
    'opt-adamw': dict(opt='adamw', lr=1e-3),
    'opt-rmsprop': dict(opt='rmsprop', lr=1e-3),
    'lr-0.01': dict(lr=0.01),                            # --- 学习率 / 权重衰减 ---
    'lr-0.1': dict(lr=0.1),
    'wd-0': dict(wd=0.0),
    'wd-5e-3': dict(wd=5e-3),
    'width-0.5': dict(width=0.5),                        # --- 容量 ---
    'width-1.5': dict(width=1.5),
    'mlp': dict(model='mlp'),                            # --- 归纳偏置，同一套配方 ---
    'mlp-adam': dict(model='mlp', opt='adam', lr=1e-3),  # ... 换成它扛得住的配方
    'no-bn': dict(bn=False),                             # --- 归一化 / dropout / 初始化 ---
    'dropout-0': dict(dropout=0.0),
    'dropout-0.5': dict(dropout=0.5),
    'init-xavier': dict(init='xavier'),
    'init-kaiming': dict(init='kaiming'),
}


def run(name, seed=0):
    cfg = {**BASE, **CONFIGS[name]}
    tag = name if seed == 0 else f'{name}-seed{seed}'
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = cl.try_gpu()
    train_iter, val_iter, _ = cl.load_data_cifar10(cfg['batch_size'], root=paths.DATA,
                                                    augment=cfg['augment'])
    net = (MLP(dropout=cfg['dropout']) if cfg['model'] == 'mlp' else
           build(cfg['model'], width=cfg['width'], dropout=cfg['dropout'],
                 bn=cfg['bn'], init=cfg['init']))
    params = sum(p.numel() for p in net.parameters())
    print(f'=== {tag}: {params / 1e6:.3f}M params, {cfg}', flush=True)
    writer = SummaryWriter(paths.TB / tag)
    animator = cl.train(net, train_iter, val_iter, cfg['epochs'], cfg['lr'], device,
                         wd=cfg['wd'], ls=cfg['ls'], opt=cfg['opt'],
                         ckpt=paths.ABLATIONS / f'{tag}.pth', writer=writer)
    m = animator.metrics
    writer.add_hparams({k: v for k, v in cfg.items() if v is not None},
                       {'hparam/val_acc': m['val_acc'], 'hparam/params': params})
    writer.close()
    path = paths.METRICS / f'{tag}.json'
    with open(path, 'w') as f:
        json.dump({'name': tag, 'seed': seed, 'config': cfg, 'params': params, **m}, f, indent=2)
    print(f'=== {tag}: best val acc {m["val_acc"]:.4f} in {m["seconds"]:.0f}s', flush=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = dict(a[2:].split('=') for a in sys.argv[1:] if a.startswith('--'))
    seed = int(flags.get('seed', 0))
    for name in args or CONFIGS:
        run(name, seed)
