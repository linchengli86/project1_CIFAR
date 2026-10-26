"""Datasets: the raw CIFAR pickles, the class-balanced split, and the DataLoader helpers."""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2

__all__ = ['CIFAR10', 'CIFAR100', 'read_cifar', 'read_cifar10', 'split_by_class',
           'kfold_indices', 'load_data_cifar10', 'load_test_cifar10', 'load_data_kfold']


def _read_batch(path, key=b'labels'):
    """One CIFAR python pickle -> (uint8 NCHW images, labels)."""
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return (torch.from_numpy(d[b'data'].reshape(-1, 3, 32, 32)),
            torch.from_numpy(np.asarray(d[key], dtype=np.int64)))


class CIFAR10(Dataset):
    """A uint8 image/label pair plus normalisation and optional augmentation.

    The file layout, the label field and the channel statistics live on the class, so
    CIFAR-100 is the same loader with four attributes overridden.
    """
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    label_key = b'labels'
    files = {'train': [f'data_batch_{i}' for i in range(1, 6)], 'test': ['test_batch']}
    num_classes = 10

    def __init__(self, X, y, augment=False, normalize=True):
        self.X, self.y = X, y
        aug = [v2.RandomCrop(32, padding=4, padding_mode='reflect'),
               v2.RandomHorizontalFlip()] if augment else []
        tail = [v2.ToDtype(torch.float32, scale=True)]
        if normalize:                      # normalize=False 复现"只做 /255"的朴素基线
            tail.append(v2.Normalize(self.mean, self.std))
        self.transform = v2.Compose(aug + tail)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.transform(self.X[i]), self.y[i]


class CIFAR100(CIFAR10):
    """The 100-fine-class variant: `train`/`test` pickles with `fine_labels`."""
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    label_key = b'fine_labels'
    files = {'train': ['train'], 'test': ['test']}
    num_classes = 100


def read_cifar(root, train, cls=CIFAR10):
    """All images/labels of one split from the original python pickles (no download)."""
    names = cls.files['train' if train else 'test']
    batches = [_read_batch(os.path.join(root, n), cls.label_key) for n in names]
    return torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])  # uint8, 153 MB


def read_cifar10(root, train):
    """Backwards-compatible alias for read_cifar(root, train, CIFAR10)."""
    return read_cifar(root, train, CIFAR10)


def split_by_class(y, n_val, num_classes=10):
    """Train/val indices with exactly `n_val` validation images per class.

    Keeping the split class-balanced means the validation accuracy is an unbiased
    estimate of the (also balanced) test accuracy, and every epoch sees the same
    number of held-out samples per class.
    """
    g = torch.Generator().manual_seed(0)
    train_idx, val_idx = [], []
    for c in range(num_classes):
        idx = (y == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        val_idx += idx[:n_val].tolist()
        train_idx += idx[n_val:].tolist()
    return train_idx, val_idx


def kfold_indices(y, k=5, num_classes=10):
    """Class-balanced k-fold splits -> list of (train_idx, val_idx), one pair per fold."""
    g = torch.Generator().manual_seed(0)
    folds = [[] for _ in range(k)]
    for c in range(num_classes):
        idx = (y == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        for fold, chunk in zip(folds, idx.chunk(k)):
            fold += chunk.tolist()
    return [(sum((f for j, f in enumerate(folds) if j != i), []), folds[i]) for i in range(k)]


def _loader(dataset, batch_size, num_workers, shuffle=False):
    # drop_last only for training, so the val/test numerators stay exact
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0, drop_last=shuffle,
                      pin_memory=True)


def load_data_cifar10(batch_size, root='CIFAR-10', num_workers=0, val_per_class=500,
                      augment=True, dataset='cifar10', normalize=True):
    """Load the on-disk CIFAR-10 (or CIFAR-100); returns (train_iter, val_iter, test_iter).

    The validation split holds out `val_per_class` images of every class (5000 of the
    50000 training images by default); `test_iter` is never touched during training.
    `augment=False` turns the random crop + flip off -- the control for "how much of the
    accuracy is the augmentation?".
    `normalize=False` skips the mean/std normalisation (keeping only /255) -- together with
    `augment=False` this reproduces the naive "raw pixels" baseline of the recipe ladder.
    `dataset='cifar100'` reads the 100-fine-class pickles instead (500 images per class,
    so 50 held out per class gives the same 5000-image validation split).
    num_workers=0 keeps everything in-process (32x32 crops are cheap and it avoids
    worker-process setup on Windows); raise it on Linux if loading shows up in the profile.
    """
    cls = CIFAR100 if dataset == 'cifar100' else CIFAR10
    X, y = read_cifar(root, True, cls)
    train_idx, val_idx = split_by_class(y, val_per_class, cls.num_classes)
    X_test, y_test = read_cifar(root, False, cls)
    return (_loader(Subset(cls(X, y, augment=augment, normalize=normalize), train_idx),
                    batch_size, num_workers, True),
            _loader(Subset(cls(X, y, normalize=normalize), val_idx), batch_size, num_workers),
            _loader(cls(X_test, y_test, normalize=normalize), batch_size, num_workers))


def load_test_cifar10(root='CIFAR-10', batch_size=256, num_workers=0, dataset='cifar10'):
    """Only the test split -- evaluation does not need to read the 50k training images."""
    cls = CIFAR100 if dataset == 'cifar100' else CIFAR10
    return _loader(cls(*read_cifar(root, False, cls)), batch_size, num_workers)


def load_data_kfold(X, y, fold, k=5, batch_size=128, num_workers=0, dataset='cifar10'):
    """The (train_iter, val_iter) pair of fold `fold` of a class-balanced k-fold split.

    With k=5 each fold trains on 4/5 of the 50k training images (40k) and validates on 10k:
    the validation set is twice the size of the fixed split's, the training set 5k smaller.
    Both are class-balanced, so the per-fold accuracy is on the same scale as the fixed split.
    """
    cls = CIFAR100 if dataset == 'cifar100' else CIFAR10
    train_idx, val_idx = kfold_indices(y, k, cls.num_classes)[fold]
    return (_loader(Subset(cls(X, y, augment=True), train_idx), batch_size, num_workers, True),
            _loader(Subset(cls(X, y), val_idx), batch_size, num_workers))
