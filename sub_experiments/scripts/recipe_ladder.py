"""The recipe ladder: start from "raw pixels, plain SGD" and add one trick at a time.

Every rung differs from the previous one by exactly one change, so the drop or gain between
two rungs is that trick's marginal contribution *in this order*. Two anchors make the ladder
interpretable:

    r0  reproduces the naive baseline of the earlier exploration (project1):
        /255 only, no augmentation, no BatchNorm, SGD lr=0.003 with a constant rate,
        no label smoothing, weight decay 5e-4.
    the last rung of `cnn` equals this project's standard protocol, so the ladder should
        land on the same number as `train.py cnn-small` (a cross-check, not a coincidence).

    python scripts/recipe_ladder.py cnn            # all rungs of the CNN ladder
    python scripts/recipe_ladder.py cnn:r3         # one rung (so rungs can run in parallel)

Results go to results/metrics/ladder-<ladder>-<rung>.json and results/tb/ladder-*.
"""
import json
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

import cifarlab as cl
from cifarlab import paths
from cifarlab.models import build

# 每个 rung 都是"相对上一级只加一件事"
BASE = dict(model='cnn-small', normalize=False, augment=False, bn=False, dropout=0.0,
            lr=0.003, schedule='const', ls=0.0, opt='sgd', wd=5e-4, init=None)

LADDERS = {
    'cnn': [
        ('r0-naive',     dict()),
        ('r1-norm',      dict(normalize=True)),
        ('r2-lr-cosine', dict(lr=0.05, schedule='cosine')),
        ('r3-augment',   dict(augment=True)),
        ('r4-ls',        dict(ls=0.1)),
        ('r5-bn',        dict(bn=True)),
        ('r6-dropout',   dict(dropout=0.3)),
    ],
    'mlp': [
        ('m0-naive',      dict(model='mlp')),
        ('m1-norm',       dict(normalize=True)),
        ('m2-dropout',    dict(dropout=0.3)),
        ('m3-ls',         dict(ls=0.1)),
        ('m4-augment',    dict(augment=True)),
        ('m5-lr-cosine',  dict(lr=0.05, schedule='cosine')),
        ('m6-adam',       dict(opt='adam', lr=1e-3)),
    ],
}

# "逐个单加"：每个 arm 只相对 BASE 改一件事，彼此独立、没有顺序依赖。
# 它回答"从零开始，哪一招最值钱"；注意各 arm 的增益不能相加——trick 之间会互相替代。
SINGLE = {
    'cnn': [
        ('s1-norm',     dict(normalize=True)),
        ('s2-lr',       dict(lr=0.05, schedule='cosine')),
        ('s3-augment',  dict(augment=True)),
        ('s4-ls',       dict(ls=0.1)),
        ('s5-bn',       dict(bn=True)),
        ('s6-dropout',  dict(dropout=0.3)),
        ('s7-xavier',   dict(init='xavier')),
        ('s8-kaiming',  dict(init='kaiming')),
    ],
    'mlp': [
        ('t1-norm',     dict(model='mlp', normalize=True)),
        ('t2-lr',       dict(model='mlp', lr=0.05, schedule='cosine')),
        ('t3-dropout',  dict(model='mlp', dropout=0.3)),
        ('t4-ls',       dict(model='mlp', ls=0.1)),
        ('t5-bn',       dict(model='mlp', bn=True)),
        ('t6-adam',     dict(model='mlp', opt='adam', lr=1e-3)),
        ('t7-xavier',   dict(model='mlp', init='xavier')),
        ('t8-kaiming',  dict(model='mlp', init='kaiming')),
        ('t9-augment',  dict(model='mlp', augment=True)),
    ],
}

# ViT 不能从"朴素基线"往上加——它在那个起点根本训不起来，所有 Δ 都会变成
# "救活它需要什么"。所以改成从它的最佳配方做减法（leave-one-out）。
VIT_FULL = dict(model='vit', normalize=True, augment=True, bn=False, dropout=0.3,
                lr=1e-3, schedule='cosine', ls=0.1, opt='adamw', wd=5e-4, init=None,
                warmup=5, clip=1.0)

LOO = {
    'vit': [
        ('v0-full',        dict()),                       # 复现 80.14 作为参照
        ('v1-no-warmup',   dict(warmup=0)),
        ('v2-no-augment',  dict(augment=False)),
        ('v3-no-ls',       dict(ls=0.0)),
        ('v4-no-norm',     dict(normalize=False)),
    ],
}

EPOCHS = {'cnn': 100, 'mlp': 100, 'vit': 150}


def rung_config(ladder, rung):
    """`cnn:r3` = cumulative; `cnn-single:s3` = one change from BASE; `vit-loo:v1` = minus one."""
    if ladder.endswith('-single'):
        base = ladder[:-len('-single')]
        for name, delta in SINGLE[base]:
            if name == rung:
                cfg = dict(BASE)
                cfg.update(delta)
                return name, cfg
        raise SystemExit(f'unknown arm {rung!r} for {ladder!r}')
    if ladder.endswith('-loo'):
        base = ladder[:-len('-loo')]
        for name, delta in LOO[base]:
            if name == rung:
                cfg = dict(VIT_FULL)
                cfg.update(delta)
                return name, cfg
        raise SystemExit(f'unknown arm {rung!r} for {ladder!r}')
    steps = LADDERS[ladder]
    cfg = dict(BASE)
    for name, delta in steps:
        cfg.update(delta)
        if name == rung:
            return name, cfg
    raise SystemExit(f'unknown rung {rung!r} for ladder {ladder!r}')


def make_model(cfg):
    """Build the arm's model; `bn`/`init` only exist on some architectures."""
    kw = dict(num_classes=10, dropout=cfg['dropout'])
    try:
        return build(cfg['model'], bn=cfg['bn'], init=cfg['init'], **kw)
    except TypeError:
        return build(cfg['model'], **kw)


def run(ladder, rung):
    name, cfg = rung_config(ladder, rung)
    tag = f'ladder-{ladder}-{name}'
    epochs = EPOCHS.get(ladder.split('-')[0], 100)
    torch.manual_seed(0)                      # 所有 rung 用同一个种子，差异只来自配方
    torch.backends.cudnn.benchmark = True
    device = cl.try_gpu()
    train_iter, val_iter, test_iter = cl.load_data_cifar10(
        128, root=paths.DATA, augment=cfg['augment'], normalize=cfg['normalize'])
    net = make_model(cfg)
    params = sum(p.numel() for p in net.parameters())
    print(f'=== {tag}: {params / 1e6:.3f}M  {cfg}', flush=True)
    writer = SummaryWriter(paths.TB / tag)
    animator = cl.train(net, train_iter, val_iter, epochs, cfg['lr'], device,
                        wd=cfg['wd'], ls=cfg['ls'], opt=cfg['opt'],
                        ckpt=paths.ABLATIONS / f'{tag}.pth', writer=writer,
                        schedule=cfg['schedule'], warmup=cfg.get('warmup', 0),
                        clip=cfg.get('clip'))
    writer.close()
    net.load_state_dict(torch.load(paths.ABLATIONS / f'{tag}.pth', map_location=device,
                                   weights_only=True))
    test = cl.evaluate(net, test_iter, device, loss_fn=torch.nn.CrossEntropyLoss())
    m = animator.metrics
    record = {'name': tag, 'ladder': ladder, 'rung': name, 'config': cfg, 'params': params,
              'val_acc': m['val_acc'], 'test_acc1': test['acc@1'], 'test_acc3': test['acc@3'],
              'seconds': m['seconds'], 'epochs': epochs}
    with open(paths.METRICS / f'{tag}.json', 'w') as f:
        json.dump(record, f, indent=2)
    print(f'=== {tag}: val {m["val_acc"]:.4f}  test {test["acc@1"]:.4f}  '
          f'({m["seconds"] / 60:.1f} min)', flush=True)
    return record


if __name__ == '__main__':
    targets = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not targets:
        raise SystemExit(__doc__)
    for target in targets:
        ladder, _, rung = target.partition(':')
        if ladder.endswith('-single'):
            arms = SINGLE.get(ladder[:-len('-single')])
        elif ladder.endswith('-loo'):
            arms = LOO.get(ladder[:-len('-loo')])
        else:
            arms = LADDERS.get(ladder)
        if arms is None:
            raise SystemExit(f'unknown ladder {ladder!r}; choose from '
                             f'{list(LADDERS) + [k + "-single" for k in SINGLE]}')
        chosen = [rung] if rung else [name for name, _ in arms]
        for one in chosen:
            run(ladder, one)
