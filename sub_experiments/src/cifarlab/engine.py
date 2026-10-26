"""Training loop, schedules and evaluation (the "d2l train_ch6" of this project)."""
import math
import os
import time

import torch
from torch import nn

__all__ = ['try_gpu', 'Timer', 'Accumulator', 'Animator', 'accuracy', 'evaluate',
           'evaluate_accuracy', 'train_epoch', 'make_optimizer', 'train']


def try_gpu(i=0):
    """Return gpu(i) if it exists, otherwise return cpu()."""
    return torch.device(f'cuda:{i}') if torch.cuda.device_count() >= i + 1 else torch.device('cpu')


class Timer:
    """Record multiple running times (d2l's Timer)."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """Accumulate values over n variables (d2l's Accumulator)."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Animator:
    """Collect metrics epoch by epoch and draw them at the end (headless d2l.Animator)."""
    def __init__(self, xlabel='epoch', ylabel='', legend=(), title=''):
        self.xlabel, self.ylabel, self.legend, self.title = xlabel, ylabel, legend, title
        self.history = []  # rows of (x, y1, y2, ..., yn)
        self.metrics = {}

    def add(self, x, *ys):
        self.history.append((x,) + ys)

    def draw(self, path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(self.legend), figsize=(3.8 * len(self.legend), 3.2),
                                 squeeze=False)
        xs = [row[0] for row in self.history]
        columns = list(zip(*[row[1:] for row in self.history]))  # one tuple per metric
        for ax, name, ys in zip(axes[0], self.legend, columns):
            ax.plot(xs, ys, linewidth=1.5)
            ax.set_xlabel(self.xlabel)
            ax.set_title(name, fontsize=10)
            ax.xaxis.get_major_locator().set_params(integer=True)
            ax.grid(alpha=0.3)
        fig.suptitle(self.title)
        fig.tight_layout()
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches='tight')
        return fig


def accuracy(y_hat, y):
    """Number of correct predictions in a minibatch."""
    return (y_hat.argmax(axis=1) == y).float().sum().item()


def evaluate(net, data_iter, device, loss_fn=None, topk=(1, 3)):
    """Loss / top-k accuracy over a dataset -> {'acc@1': .., 'acc@3': .., 'loss': ..}."""
    net.eval()
    n, loss_sum, correct = 0, 0.0, [0] * len(topk)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_hat = net(X)
            if loss_fn is not None:
                loss_sum += loss_fn(y_hat, y).item() * y.numel()
            pred = y_hat.topk(max(topk), dim=1).indices
            for j, k in enumerate(topk):
                correct[j] += (pred[:, :k] == y[:, None]).any(dim=1).sum().item()
            n += y.numel()
    out = {f'acc@{k}': c / n for k, c in zip(topk, correct)}
    if loss_fn is not None:
        out['loss'] = loss_sum / n
    return out


def evaluate_accuracy(net, data_iter, device):
    """Top-1 accuracy of `net` on `data_iter`."""
    return evaluate(net, data_iter, device, topk=(1,))['acc@1']


def train_epoch(net, train_iter, loss, updater, device, clip=None):
    """One pass over the training set; returns (mean loss, accuracy)."""
    net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        if clip:                                  # transformers need this on the early steps
            nn.utils.clip_grad_norm_(net.parameters(), clip)
        updater.step()
        metric.add(l.item() * y.numel(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def make_optimizer(name, params, lr, wd):
    """The optimizer menu for the 'try different optimizer' experiment (cosine is added by train)."""
    opts = {'sgd': lambda: torch.optim.SGD(params, lr, momentum=0.9, weight_decay=wd),
            'adam': lambda: torch.optim.Adam(params, lr, weight_decay=wd),
            'adamw': lambda: torch.optim.AdamW(params, lr, weight_decay=wd),
            'rmsprop': lambda: torch.optim.RMSprop(params, lr, momentum=0.9, weight_decay=wd)}
    return opts[name]()


def train(net, train_iter, val_iter, num_epochs, lr, device, wd=5e-4, ls=0.1, opt='sgd',
          ckpt=None, writer=None, warmup=0, clip=None, schedule='cosine'):
    """d2l's train_ch6: cosine schedule + label smoothing, keeps the best-on-val weights.

    `warmup` linearly ramps the learning rate over that many epochs before the cosine
    decay starts (0 = the plain cosine used by the CNN runs). `clip` caps the global
    gradient norm. Both matter for transformers, and neither changes the default path.
    `schedule='const'` keeps the learning rate fixed instead of decaying -- only used by
    the recipe ladder to reproduce a naive constant-lr baseline; the default is unchanged.
    `writer` is a torch.utils.tensorboard.SummaryWriter: when given, loss/accuracy/
    learning rate are logged for every epoch, so runs can be compared in TensorBoard.
    """
    net.to(device)
    if ckpt:
        os.makedirs(os.path.dirname(ckpt) or '.', exist_ok=True)
    loss = nn.CrossEntropyLoss(label_smoothing=ls)
    eval_loss = nn.CrossEntropyLoss()
    updater = make_optimizer(opt, net.parameters(), lr, wd)

    def lr_lambda(epoch):                     # multiplier on the base lr
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        if schedule == 'const':
            return 1.0
        t = (epoch - warmup) / max(1, num_epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(updater, lr_lambda)
    animator = Animator('epoch', '', ('train loss', 'train acc', 'val acc'), net.__class__.__name__)
    timer, best, best_epoch = Timer(), 0.0, 0
    print(f'{"epoch":>5} {"lr":>7} {"train loss":>10} {"train acc":>9} {"val acc":>8} {"sec":>7}')
    for epoch in range(1, num_epochs + 1):
        timer.start()
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater, device, clip)
        val = evaluate(net, val_iter, device, loss_fn=eval_loss)
        animator.add(epoch, train_loss, train_acc, val['acc@1'])
        if writer:
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val['loss'], epoch)
            writer.add_scalar('acc/train', train_acc, epoch)
            writer.add_scalar('acc/val@1', val['acc@1'], epoch)
            writer.add_scalar('acc/val@3', val['acc@3'], epoch)
            writer.add_scalar('lr', updater.param_groups[0]['lr'], epoch)
        print(f'{epoch:5d} {updater.param_groups[0]["lr"]:7.4f} {train_loss:10.4f} '
              f'{train_acc:9.4f} {val["acc@1"]:8.4f} {timer.stop():6.1f}s', flush=True)
        if val['acc@1'] > best:
            best, best_epoch = val['acc@1'], epoch
            if ckpt:
                torch.save(net.state_dict(), ckpt)
        scheduler.step()
    animator.metrics = {'val_acc': best, 'val_acc_last': animator.history[-1][-1],
                        'val_acc3_last': val['acc@3'], 'best_epoch': best_epoch,
                        'seconds': timer.sum(), 'epochs': num_epochs, 'lr': lr, 'opt': opt}
    return animator
