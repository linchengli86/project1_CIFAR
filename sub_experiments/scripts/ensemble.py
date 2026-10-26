"""把若干模型平均起来（soft voting），在同一测试集上比较单体与组合。

    python ensemble.py                      # 默认四个 ~1M 参数的模型
    python ensemble.py googlenet resnet     # 只组合这几个

每个模型都取"验证集最优"的权重，组合方式就是 softmax 的等权平均；另外报告一个
oracle 上界（任意一个模型答对就算对），用来看各模型到底有多互补。
"""
import json
import sys

import torch

import cifarlab as cl
from cifarlab import paths
from cifarlab.models import DEFAULT_ENSEMBLE, build


def main():
    names = sys.argv[1:] or DEFAULT_ENSEMBLE
    device = cl.try_gpu()
    nets = []
    for name in names:
        net = build(name).to(device)
        net.load_state_dict(torch.load(paths.CKPT / f'{name}_best.pth', map_location=device,
                                       weights_only=True))
        nets.append(net.eval())
    test_iter = cl.load_test_cifar10(paths.DATA)
    single, ens, oracle, total = {n: 0 for n in names}, 0, 0, 0
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            probs = [torch.softmax(net(X), dim=1) for net in nets]
            for name, p in zip(names, probs):
                single[name] += (p.argmax(1) == y).sum().item()
            ens += (torch.stack(probs).mean(0).argmax(1) == y).sum().item()
            oracle += torch.stack([p.argmax(1) == y for p in probs]).any(0).sum().item()
            total += y.numel()
    for name in names:
        print(f'{name:10s} {single[name] / total:.4f}')
    print(f'{"ensemble":10s} {ens / total:.4f}  ({len(names)} models, soft voting)')
    print(f'{"oracle":10s} {oracle / total:.4f}  (any model correct)')
    with open(paths.METRICS / 'ensemble.json', 'w') as f:
        json.dump({'models': names, 'single': {n: single[n] / total for n in names},
                   'ensemble': ens / total, 'oracle': oracle / total}, f, indent=2)


if __name__ == '__main__':
    main()
