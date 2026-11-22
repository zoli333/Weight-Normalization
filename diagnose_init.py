"""Why does the data-dependent init start *slower* than the plain init (epoch-1 loss 2.9 vs 1.8 in the old results)?

Builds one model exactly like train.py, then prints
  1) the per-layer output statistics on a fresh training batch (right after the init) - with the data-dependent
     init, every weight layer should output ~0 mean and ~1 std per unit,
  2) the range of the weight-norm gains g (g = 1/sigma can be large for deep layers),
  3) the training loss of the first optimizer steps (chance level is ln(10) = 2.303).

Things worth comparing (each one is a separate run):
    python diagnose_init.py --method weight_norm_with_init
    python diagnose_init.py --method weight_norm_with_init --init-mode eval     # no noise/dropout during the init
    python diagnose_init.py --method weight_norm_with_init --lr 0.001           # smaller Adam steps
    python diagnose_init.py --method weight_norm                                # reference without init
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from layers import WeightNormLayer, MeanOnlyBatchNormLayer, NINLayer
from model import device
from train import METHODS, get_datasets, build_model

REPORTED = (MeanOnlyBatchNormLayer, WeightNormLayer, nn.Conv2d, nn.Linear, NINLayer, nn.BatchNorm1d, nn.BatchNorm2d)


def print_layer_stats(model, x):
    rows, handles = [], []

    def make_hook(idx, name):
        def hook(module, inputs, out):
            dims = [d for d in range(out.dim()) if d != 1]
            ch_mean, ch_std = out.mean(dim=dims), out.std(dim=dims)
            rows.append((idx, name, ch_mean.abs().mean().item(), ch_std.mean().item(),
                         ch_std.min().item(), ch_std.max().item()))
        return hook

    for idx, module in enumerate(model.model):
        if isinstance(module, REPORTED):
            handles.append(module.register_forward_hook(make_hook(idx, type(module).__name__)))
    model.train()
    with torch.no_grad():
        logits = model(x)
    for h in handles:
        h.remove()

    print('\nper-layer output statistics on a fresh batch (train mode):')
    print(f'{"idx":>3} {"layer":<24} {"mean|ch mean|":>14} {"mean ch std":>12} {"min ch std":>11} {"max ch std":>11}')
    for r in rows:
        print(f'{r[0]:>3} {r[1]:<24} {r[2]:>14.4f} {r[3]:>12.4f} {r[4]:>11.4f} {r[5]:>11.4f}')
    return logits


def print_gains(model):
    print('\nweight-norm gains g (per layer: min / median / max):')
    for idx, module in enumerate(model.model):
        wn = module if isinstance(module, WeightNormLayer) else getattr(module, 'module', None)
        if isinstance(wn, WeightNormLayer):
            g = wn.module.g.detach().flatten()
            print(f'{idx:>3} {g.min().item():>10.3f} {g.median().item():>10.3f} {g.max().item():>10.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=list(METHODS), default='weight_norm_with_init')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--init-mode', choices=['train', 'eval'], default='train')
    p.add_argument('--noise', choices=['additive', 'relative'], default='additive')
    p.add_argument('--data', default='./data')
    args = p.parse_args()

    cfg = METHODS[args.method]
    lr = args.lr if args.lr is not None else cfg['lr']
    _, trainset_plain, _ = get_datasets(args.data, augment=False)
    model = build_model(cfg, args.seed, trainset_plain, noise_mode=args.noise, init_mode=args.init_mode)

    g = torch.Generator()
    g.manual_seed(12345)
    batches = iter(DataLoader(trainset_plain, batch_size=100, shuffle=True, generator=g))
    x, y = next(batches)
    x, y = x.to(device), y.to(device)

    print(f'method={args.method} lr={lr:g} init_mode={args.init_mode} noise={args.noise} seed={args.seed}')
    logits = print_layer_stats(model, x)
    print_gains(model)
    print(f'\ninitial loss on that batch: {F.cross_entropy(logits, y).item():.4f}  (chance level: 2.3026)')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    window = []
    print('\nfirst optimizer steps (mean loss over windows of 20 steps):')
    for step in range(1, args.steps + 1):
        try:
            x, y = next(batches)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        window.append(loss.item())
        if step % 20 == 0:
            print(f'  steps {step - 19:>4}-{step:<4} loss {sum(window) / len(window):.4f}')
            window = []


if __name__ == '__main__':
    main()
