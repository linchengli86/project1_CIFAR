"""主实验：在本地 CIFAR-10/100 上从头训练，四个模型用完全相同的配方。

    python train.py                      # 默认：小型 GoogLeNet，seed 42，固定验证集
    python train.py resnet               # 换模型
    python train.py vgg-res              # VGG + 残差
    python train.py googlenet 0          # 换种子（划分固定，只重抽初始化与 batch 顺序）
    python train.py vit 42 adamw --epochs=150 --warmup=5 --clip=1.0
    python train.py resnet --dataset=cifar100

两种验证协议：
    固定划分（默认）  45k 训练 / 5k 验证，一次训练一个模型
    `--cv=K`          每折 40k 训练 / 10k 验证，跑 K 次，报 K 折均值 ± 标准差
                      每折用 seed+f 重新初始化，汇总写到 results/metrics/<tag>.json

每折完全独立（划分 / 种子 / 权重 / 日志都不共享），所以可以并行：
    `--jobs=N` 同时跑 N 折（默认 1，即串行；N=K 就是全并行）。
    `--fold=i` 只跑第 i 折（并行时由父进程调用，也可单独调试）。

产物：results/checkpoints/<tag>_best.pth、results/figures/<tag>_curves.png、
results/tb/<tag>、results/metrics/<tag>.json。
tag 在 seed=42 时就是模型名，其余种子为 `<model>-seed<seed>`，互不覆盖。
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import cifarlab as cl
from cifarlab import paths
from cifarlab.models import MODELS, build

# 128 x 0.05 是 256 x 0.1 配方在 CIFAR-10 上的等比缩放
BATCH_SIZE, EPOCHS, LR = 128, 100, 0.05
VAL_PER_CLASS = 500  # 每类留 500 张做验证，共 5000 张
DEFAULT_SEED = 42


def run(name, tag, seed, train_iter, val_iter, test_iter, n_classes, epochs, lr, opt,
        warmup, clip, device, dataset='cifar10'):
    """Train one model on one split, evaluate on test, and record the metrics."""
    torch.manual_seed(seed)                      # 必须在建模型之前，初始化才可复现
    net = build(name, num_classes=n_classes)
    params = sum(p.numel() for p in net.parameters())
    print(f'{tag}: {params / 1e6:.3f}M params, seed {seed}, opt {opt}, epochs {epochs}, '
          f'warmup {warmup}, clip {clip}, {dataset}, {device}, '
          f'{len(train_iter.dataset)} train / {len(val_iter.dataset)} val / '
          f'{len(test_iter.dataset)} test', flush=True)
    ckpt = paths.CKPT / f'{tag}_best.pth'
    writer = SummaryWriter(paths.TB / tag)
    animator = cl.train(net, train_iter, val_iter, epochs, lr, device, opt=opt,
                        ckpt=ckpt, writer=writer, warmup=warmup, clip=clip)
    writer.close()
    animator.draw(paths.FIGURES / f'{tag}_curves.png')
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    metrics = cl.evaluate(net, test_iter, device, loss_fn=torch.nn.CrossEntropyLoss())
    print(f'best val acc: {animator.metrics["val_acc"]:.4f}')
    print(f'test acc@1: {metrics["acc@1"]:.4f}  acc@3: {metrics["acc@3"]:.4f} '
          f'(weights selected on val)')
    record = {'name': tag, 'model': name, 'seed': seed, 'params': params,
              'val_acc': animator.metrics['val_acc'],
              'test_acc1': metrics['acc@1'], 'test_acc3': metrics['acc@3'],
              'seconds': animator.metrics['seconds']}
    with open(paths.METRICS / f'{tag}.json', 'w') as f:
        json.dump(record, f, indent=2)
    return record


def run_fold(name, tag, seed, root, dataset, n_classes, k, fold, epochs, lr, opt, warmup,
             clip, device):
    """Load fold `fold` of `k` and train on it."""
    cls = cl.CIFAR100 if dataset == 'cifar100' else cl.CIFAR10
    X, y = cl.read_cifar(root, True, cls)
    test_iter = cl.load_test_cifar10(root, batch_size=256, dataset=dataset)
    train_iter, val_iter = cl.load_data_kfold(X, y, fold, k=k, batch_size=BATCH_SIZE,
                                              dataset=dataset)
    return run(name, f'{tag}-f{fold}', seed + fold, train_iter, val_iter, test_iter,
               n_classes, epochs, lr, opt, warmup, clip, device, dataset)


def aggregate(tag, k, model, seed, epochs):
    """Collect the per-fold records written by run_fold and summarise them."""
    rows = []
    for fold in range(k):
        with open(paths.METRICS / f'{tag}-f{fold}.json') as f:
            record = json.load(f)
        record['fold'] = fold
        rows.append(record)
    accs = [r['test_acc1'] for r in rows]
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f'\n{k}-fold test acc: {mean:.4f} +- {std:.4f}  (folds: '
          + ', '.join(f'{a:.4f}' for a in accs) + ')')
    with open(paths.METRICS / f'{tag}.json', 'w') as f:
        json.dump({'name': tag, 'model': model, 'k': k, 'epochs': epochs, 'seed': seed,
                   'folds': rows, 'test_acc1_mean': mean, 'test_acc1_std': std,
                   'val_acc_mean': statistics.mean(r['val_acc'] for r in rows)}, f, indent=2)
    return rows


def spawn_folds(name, tag, seed, opt, dataset, k, epochs, warmup, clip, jobs):
    """Run the folds as independent processes, at most `jobs` at a time."""
    script = str(Path(__file__).resolve())
    running = []
    started = time.time()
    for fold in range(k):
        cmd = [sys.executable, script, name, str(seed), opt, f'--cv={k}', f'--fold={fold}',
               f'--epochs={epochs}', f'--dataset={dataset}']
        if warmup:
            cmd.append(f'--warmup={warmup}')
        if clip is not None:
            cmd.append(f'--clip={clip}')
        log = open(paths.LOGS / f'{tag}-f{fold}.txt', 'w')
        running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        while sum(p.poll() is None for p, _ in running) >= jobs:
            time.sleep(1)
    codes = [p.wait() for p, _ in running]
    for _, log in running:
        log.close()
    print(f'{k} folds finished in {time.time() - started:.0f}s  (exit codes: {codes})')
    if any(codes):
        raise SystemExit('some folds failed; see results/logs/')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = dict(a[2:].split('=') for a in sys.argv[1:] if a.startswith('--'))
    name = args[0] if args else 'googlenet'
    if name not in MODELS and name != 'vgg-res':
        raise SystemExit(f'unknown model {name!r}; choose from {list(MODELS) + ["vgg-res"]}')
    seed = int(args[1]) if len(args) > 1 else DEFAULT_SEED
    opt = args[2] if len(args) > 2 else 'sgd'
    default_lr = LR if opt == 'sgd' else 1e-3   # 自适应方法用 1e-3，与 experiments.py 一致
    lr = float(flags['lr']) if 'lr' in flags else default_lr
    epochs = int(flags.get('epochs', EPOCHS))
    warmup = int(flags.get('warmup', 0))
    clip = float(flags['clip']) if 'clip' in flags else None
    dataset = flags.get('dataset', 'cifar10')
    cv = int(flags.get('cv', 0))
    fold = int(flags['fold']) if 'fold' in flags else None
    jobs = int(flags.get('jobs', 1))
    n_classes = 10 if dataset == 'cifar10' else 100
    root = paths.DATA if dataset == 'cifar10' else paths.DATA100
    val_per_class = VAL_PER_CLASS if dataset == 'cifar10' else 50   # 500*100 与 50*10 都是 5000
    tag = name if seed == DEFAULT_SEED else f'{name}-seed{seed}'
    if opt != 'sgd':
        tag += f'-{opt}'
    if lr != default_lr:
        tag += f'-lr{lr:g}'
    if warmup:
        tag += f'-w{warmup}'
    if epochs != EPOCHS:
        tag += f'-e{epochs}'
    if dataset != 'cifar10':
        tag += '-c100'
    if cv:
        tag += f'-cv{cv}'
    if fold is not None:
        if not cv:
            raise SystemExit('--fold requires --cv')
        tag += f'-f{fold}'

    torch.backends.cudnn.benchmark = True
    device = cl.try_gpu()

    if fold is not None:                        # 子进程：只跑一折
        run_fold(name, tag.rsplit(f'-f{fold}', 1)[0], seed, root, dataset, n_classes, cv,
                 fold, epochs, lr, opt, warmup, clip, device)
        return

    if cv:
        if jobs > 1:
            spawn_folds(name, tag, seed, opt, dataset, cv, epochs, warmup, clip, jobs)
        else:
            for f in range(cv):
                run_fold(name, tag, seed, root, dataset, n_classes, cv, f, epochs, lr, opt,
                         warmup, clip, device)
                print(f'--- fold {f} done ---\n', flush=True)
        aggregate(tag, cv, name, seed, epochs)
        return

    train_iter, val_iter, test_iter = cl.load_data_cifar10(
        BATCH_SIZE, root=root, val_per_class=val_per_class, dataset=dataset)
    run(name, tag, seed, train_iter, val_iter, test_iter, n_classes, epochs, lr, opt,
        warmup, clip, device, dataset)


if __name__ == '__main__':
    main()
