"""CLIP zero-shot and 3-epoch fine-tuning, evaluated on this project's test split.

The point of this script is a ceiling, not a benchmark entry: CLIP arrives pretrained, so it
answers "what does external prior buy" while everything else here is trained from
scratch. The fine-tuning recipe follows the earlier exploration (AdamW 1e-6, 3 epochs, batch
16, one fixed prompt per class); only the evaluation changes to this project's split and test
set, so the numbers sit next to the from-scratch table honestly.

    python scripts/clip_eval.py                     # CIFAR-10, ViT-B/32
    python scripts/clip_eval.py --model=RN50
    python scripts/clip_eval.py --dataset=cifar100

Needs the `clip` package and the official weights (`ViT-B-32.pt` / `RN50.pt`), which are not
part of this repository. Point `--clip-root` at the directory that holds them.
"""
import json
import pickle
import sys

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode

import cifarlab as cl
from cifarlab import paths

CLASSES10 = ('airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck')
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPView(Dataset):
    """Raw uint8 CIFAR images -> CLIP's 224x224 bicubic-normalised view."""

    def __init__(self, X, y):
        self.X, self.y = X, y
        self.tf = v2.Compose([
            v2.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(self.X[i]), self.y[i]


@torch.no_grad()
def text_features(net, text):
    """Encode the prompts once; clip() would otherwise re-run the text tower every batch."""
    feat = net.encode_text(text).float()
    return feat / feat.norm(dim=-1, keepdim=True)


def logits_of(net, X, text_feat):
    """Same arithmetic as clip(X, text)[0], minus the repeated text encoding."""
    image = net.encode_image(X).float()
    image = image / image.norm(dim=-1, keepdim=True)
    return net.logit_scale.exp() * image @ text_feat.t()


@torch.no_grad()
def accuracy(net, loader, text_feat, device):
    net.eval()
    correct = total = 0
    for X, y in loader:
        logits = logits_of(net, X.to(device), text_feat)
        correct += (logits.argmax(1).cpu() == y).sum().item()
        total += y.numel()
    return correct / total


def prompts_for(dataset):
    """One fixed prompt per class; CIFAR-100's 100 fine names come from the meta pickle."""
    if dataset == 'cifar10':
        names = list(CLASSES10)
    else:
        with open(paths.DATA100 / 'meta', 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
        names = [n.decode().replace('_', ' ') for n in meta[b'fine_label_names']]
    return [f'a photo of a {name}.' for name in names]


def main():
    flags = dict(a[2:].split('=') for a in sys.argv[1:] if a.startswith('--'))
    model_name = flags.get('model', 'ViT-B/32')
    dataset = flags.get('dataset', 'cifar10')
    clip_root = flags.get('clip_root')
    if not clip_root:
        raise SystemExit(
            'CLIP 权重是第三方资源（ViT-B-32.pt 338 MB / RN50.pt 244 MB），不随本仓库分发。\n'
            '下载后用 --clip-root=<目录> 指过去，例如\n'
            '    python scripts/clip_eval.py --clip-root=D:/weights/clip')
    device = cl.try_gpu()
    cls = cl.CIFAR100 if dataset == 'cifar100' else cl.CIFAR10
    root = paths.DATA100 if dataset == 'cifar100' else paths.DATA
    val_per_class = 50 if dataset == 'cifar100' else 500     # both give a 5k validation set
    tag = 'clip-' + model_name.lower().replace('/', '').replace('-', '')
    if dataset != 'cifar10':
        tag += '-c100'

    import clip
    net, _ = clip.load(model_name, device='cpu', jit=False, download_root=clip_root)
    net = net.float().to(device)                    # 官方权重是 fp16，训练用 fp32
    prompts = prompts_for(dataset)
    text = clip.tokenize(prompts).to(device)
    print(f'{tag}: {len(prompts)} prompts, first = {prompts[0]!r}', flush=True)

    X, y = cl.read_cifar(root, True, cls)
    train_idx, val_idx = cl.split_by_class(y, val_per_class, cls.num_classes)
    X_test, y_test = cl.read_cifar(root, False, cls)
    view = CLIPView(X, y)
    train_iter = DataLoader(Subset(view, train_idx), batch_size=16, shuffle=True)
    val_iter = DataLoader(Subset(view, val_idx), batch_size=64)
    test_iter = DataLoader(CLIPView(X_test, y_test), batch_size=64)

    text_feat = text_features(net, text)
    zero = accuracy(net, test_iter, text_feat, device)
    print(f'{tag}: zero-shot test acc {zero:.4f}', flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=1e-6, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    best = {'epoch': 0, 'val': zero, 'test': zero}
    history = []
    for epoch in range(1, 4):
        net.train()
        running = 0.0
        for Xb, yb in train_iter:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = loss_fn(logits_of(net, Xb, text_feat), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running = loss.item()
        val = accuracy(net, val_iter, text_feat, device)
        test = accuracy(net, test_iter, text_feat, device)
        history.append({'epoch': epoch, 'loss': running, 'val_acc': val, 'test_acc1': test})
        print(f'{tag}: epoch {epoch}  val {val:.4f}  test {test:.4f}', flush=True)
        if val > best['val']:
            best = {'epoch': epoch, 'val': val, 'test': test}

    print(f'{tag}: best epoch {best["epoch"]}  val {best["val"]:.4f}  test {best["test"]:.4f}')
    with open(paths.METRICS / f'{tag}.json', 'w') as f:
        json.dump({'name': tag, 'model': model_name, 'dataset': dataset, 'pretrained': True,
                   'epochs': 3, 'batch_size': 16, 'lr': 1e-6, 'prompts': prompts[:3],
                   'zero_shot_test': zero, 'best': best, 'history': history,
                   'params': sum(p.numel() for p in net.parameters()),
                   'note': 'fine-tuned on the train split, selected on val, evaluated on test'},
                  f, indent=2)


if __name__ == '__main__':
    main()
