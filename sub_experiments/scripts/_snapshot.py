"""Freeze the numeric behaviour of the current tree so a refactor can be checked against it.

    python _snapshot.py save   # before the move  -> _snapshot.json
    python _snapshot.py check  # after the move   -> compares, prints PASS/FAIL
"""
import hashlib
import json
import sys

import torch

import cifarlab as cl
from cifarlab.models import build

MODELS = ['googlenet', 'resnet', 'vgg', 'vgg-res', 'vit', 'mlp']


def model_hash(name):
    torch.manual_seed(42)
    net = build(name)
    sd = net.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    # 固定输入下的前向输出也必须一致
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    net.eval()
    with torch.no_grad():
        out = net(x)
    h.update(out.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def split_hash():
    from cifarlab import paths
    X, y = cl.read_cifar10(paths.DATA, True)
    tr, va = cl.split_by_class(y, 500)
    h = hashlib.sha256()
    h.update(json.dumps(tr).encode())
    h.update(json.dumps(va).encode())
    return h.hexdigest()


def step_loss():
    """One SGD step on a fixed batch: catches any change in loss, init or optimizer order."""
    torch.manual_seed(42)
    net = build('googlenet')
    torch.manual_seed(1)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for _ in range(3):
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
    return round(loss.item(), 6)


def snapshot():
    return {
        'models': {m: model_hash(m) for m in MODELS},
        'split': split_hash(),
        'step_loss': step_loss(),
        'params': {m: sum(p.numel() for p in build(m).parameters()) for m in MODELS},
    }


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'save'
    snap = snapshot()
    if mode == 'save':
        with open('_snapshot.json', 'w') as f:
            json.dump(snap, f, indent=1)
        print('saved')
        for m, h in snap['models'].items():
            print(f'  {m:10s} {snap["params"][m]:8d}  {h[:16]}')
        print('  split', snap['split'][:16], '| step_loss', snap['step_loss'])
    else:
        old = json.load(open('_snapshot.json'))
        ok = True
        for m in MODELS:
            same = old['models'][m] == snap['models'][m]
            samep = old['params'][m] == snap['params'][m]
            ok &= same and samep
            print(f'  {m:10s} weights {"OK" if same else "CHANGED"}  '
                  f'params {"OK" if samep else "CHANGED"}')
        s = old['split'] == snap['split']
        l = abs(old['step_loss'] - snap['step_loss']) < 1e-6
        ok &= s and l
        print(f'  split {"OK" if s else "CHANGED"} | step_loss '
              f'{old["step_loss"]} -> {snap["step_loss"]} {"OK" if l else "CHANGED"}')
        print('PASS' if ok else 'FAIL')
