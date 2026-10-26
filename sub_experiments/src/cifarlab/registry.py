"""The four extension points of this project, in one place.

Nothing here is used to produce the reported numbers — it is a thin, read-only view that
makes the knobs discoverable.  Every entry already existed as a function argument or a
module-level dict; this module only names them.

    model       cifarlab.models.MODELS      build(name, **kw)
    dataset     load_data_cifar10(..., dataset='cifar10' | 'cifar100')
    optimiser   cifarlab.engine.make_optimizer(name, params, lr, wd)
    schedule    cifarlab.engine.train(..., warmup=E, clip=C)

To add a model: drop a file in `models/` that subclasses `torch.nn.Module` and register it
in `models/__init__.py`.  Nothing else needs to change.
"""
from .data import CIFAR10, CIFAR100
from .models import MODELS, build

__all__ = ['MODELS', 'build', 'DATASETS', 'OPTIMIZERS', 'SCHEDULES', 'describe']

DATASETS = {'cifar10': CIFAR10, 'cifar100': CIFAR100}
OPTIMIZERS = ('sgd', 'adam', 'adamw', 'rmsprop')
SCHEDULES = ('cosine', 'cosine+warmup')


def describe():
    """Print everything that can be swapped, with its parameter cost."""
    import torch

    from .models import build as _build

    print('models')
    for name in MODELS:
        net = _build(name)
        n = sum(p.numel() for p in net.parameters())
        print(f'  {name:10s} {n / 1e6:6.3f} M  {net.__class__.__name__}')
        del net
    torch.cuda.empty_cache()
    print('datasets')
    for name, cls in DATASETS.items():
        print(f'  {name:10s} {cls.num_classes:3d} classes  files={list(cls.files)}')
    print('optimisers:', ', '.join(OPTIMIZERS))
    print('schedules :', ', '.join(SCHEDULES))


if __name__ == '__main__':
    describe()
