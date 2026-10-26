"""cifarlab — a small, explicit training framework for CIFAR-10/100.

    import cifarlab as cl
    cl.build('resnet')                 # any registered model
    cl.load_data_cifar10(128)          # class-balanced train/val/test loaders
    cl.train(net, tr, va, 100, 0.05, cl.try_gpu())

Layout
    data.py     raw pickles -> tensors -> class-balanced split -> loaders
    engine.py   train loop, cosine/warmup schedule, optimisers, evaluation
    models/     one file per architecture, registered in models/__init__.py
    paths.py    every on-disk location the project uses
"""
from .data import *      # noqa: F401,F403
from .engine import *    # noqa: F401,F403
from .models import MODELS, build  # noqa: F401

__all__ = ['MODELS', 'build']
