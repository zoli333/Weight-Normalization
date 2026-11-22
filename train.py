import os
# Must be set before CUDA is initialized (needed for deterministic cuBLAS); no need to pass it on the command line anymore.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Model, device

# name -> configuration. lr: 0.003 for everything, except the un-normalized reference net (unstable at 0.003).
METHODS = {
    'no_norm': dict(normalizer='no_norm', init='gaussian', lr=0.0003, sample_batch_size=0),
    'weight_norm': dict(normalizer='weight', init='gaussian', lr=0.003, sample_batch_size=0),
    'weight_norm_with_init': dict(normalizer='weight', init='gaussian_datadep', lr=0.003, sample_batch_size=500),
    'batch_norm': dict(normalizer='batch_norm', init='gaussian', lr=0.003, sample_batch_size=0),
    'weight_norm_and_mean_only_batch_norm': dict(normalizer='weight_mean_only_batch_norm', init='gaussian',
                                                 lr=0.003, sample_batch_size=0),
    'weight_norm_and_mean_only_batch_norm_with_init': dict(normalizer='weight_mean_only_batch_norm',
                                                           init='gaussian_datadep', lr=0.003, sample_batch_size=500),
}

CSV_FIELDS = ['epoch', 'lr', 'train_loss_running', 'train_loss_eval', 'train_acc_eval', 'test_loss', 'test_acc',
              'elapsed_s']


def set_deterministic_mode(seed):
    """Seed every RNG and ask PyTorch for deterministic algorithms (warn, instead of fail, if one is missing)."""
    torch.manual_seed(seed)  # seeds the CPU and all CUDA devices
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_datasets(root, augment=True):
    """Returns (trainset, trainset_plain, testset).
    trainset is augmented (random +-2 px translation, horizontal flip) when augment=True;
    trainset_plain never is: it is used for the init batch and for measuring the clean train accuracy."""
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # pixel range -> [-1, 1]
    plain_tf = transforms.Compose([transforms.ToTensor(), normalize])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=2, padding_mode='reflect'),  # translation by up to 2 pixels
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = plain_tf
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    trainset_plain = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=plain_tf)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=plain_tf)
    return trainset, trainset_plain, testset


def make_sample_batch(trainset_plain, size, seed):
    g = torch.Generator()
    g.manual_seed(1000 + seed)
    idx = torch.randperm(len(trainset_plain), generator=g)[:size].tolist()
    return torch.stack([trainset_plain[i][0] for i in idx]).to(device)


def build_model(cfg, seed, trainset_plain, noise_mode='additive', init_mode='train', datadep_last_layer=True):
    # Re-seed right before the model is created: the initial weights then depend only on `seed`,
    # not on how many models were trained before in the same process.
    set_deterministic_mode(seed)
    sample_batch = None
    if cfg['sample_batch_size'] > 0:
        sample_batch = make_sample_batch(trainset_plain, cfg['sample_batch_size'], seed)
    return Model(normalizer=cfg['normalizer'], init=cfg['init'], sample_batch=sample_batch,
                 noise_mode=noise_mode, init_mode=init_mode, datadep_last_layer=datadep_last_layer).to(device)


def lr_factor(epoch, epochs, decay_start):
    """Constant until decay_start, then linear decay towards 0 (as in the paper: second half of training)."""
    if epoch < decay_start:
        return 1.0
    return max(0.0, (epochs - epoch) / (epochs - decay_start))


def warmup_factor(step, warmup_steps):
    """Linear lr warm-up over the first `warmup_steps` optimizer steps (0 = no warm-up)."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


@torch.no_grad()
def evaluate(model, loader):
    """Returns (mean loss, accuracy in %) in eval mode (no noise, no dropout, running means for normalization)."""
    was_training = model.training
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss_sum += F.cross_entropy(outputs, labels, reduction='sum').item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    model.train(was_training)
    return loss_sum / total, 100.0 * correct / total


def run_experiment(name, cfg, seed, args, datasets):
    trainset, trainset_plain, testset = datasets
    lr = args.lr if args.lr is not None else cfg['lr']
    run_dir = Path(args.out) / f'{name}__lr{lr:g}' / f'seed{seed}'
    if (run_dir / 'DONE').exists() and not args.force:
        print(f'[skip] {run_dir} is already done (use --force to rerun)')
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for marker in ('DONE', 'DIVERGED'):
        (run_dir / marker).unlink(missing_ok=True)

    model = build_model(cfg, seed, trainset_plain, noise_mode=args.noise, init_mode=args.init_mode,
                        datadep_last_layer=(args.init_last_layer == 'data'))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    decay_start = args.epochs if args.lr_decay == 'none' else (
        args.decay_start if args.decay_start is not None else args.epochs // 2)
    shuffle_generator = torch.Generator()
    loader_kwargs = dict(num_workers=args.workers, pin_memory=(device.type == 'cuda'))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, generator=shuffle_generator,
                             worker_init_fn=seed_worker, **loader_kwargs)
    steps_per_epoch = len(trainloader)
    # the scheduler is stepped after every optimizer step: warm-up (steps) x linear decay (fractional epochs)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_factor(step, args.warmup_steps)
        * lr_factor(step / steps_per_epoch, args.epochs, decay_start))
    train_eval_loader = DataLoader(trainset_plain, batch_size=500, shuffle=False, **loader_kwargs)
    testloader = DataLoader(testset, batch_size=500, shuffle=False, **loader_kwargs)

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(dict(method=name, seed=seed, decay_start=decay_start, **{**cfg, 'lr': lr}, args=vars(args)),
                  f, indent=2)

    tag = f'[{name} lr={lr:g} seed={seed}]'
    begin_time = time.time()
    diverged = False
    with open(run_dir / 'metrics.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_FIELDS)

        for epoch in range(args.epochs):
            shuffle_generator.manual_seed(seed * 1_000_003 + epoch)  # same example order for every method
            model.train()
            epoch_lr = optimizer.param_groups[0]['lr']
            running_loss = torch.zeros((), device=device)
            for inputs, labels in trainloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(inputs), labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.detach()

            train_loss = running_loss.item() / len(trainloader)  # noisy: measured with noise/dropout/augmentation on
            current_lr = epoch_lr  # lr at the first step of this epoch

            row = dict.fromkeys(CSV_FIELDS, '')
            row.update(epoch=epoch + 1, lr=current_lr, train_loss_running=train_loss,
                       elapsed_s=int(time.time() - begin_time))
            message = f'{tag} epoch {epoch + 1}/{args.epochs} lr {current_lr:.5f} train loss {train_loss:.4f}'

            if not math.isfinite(train_loss):
                diverged = True
                message += '  -> DIVERGED, stopping this run'
            elif epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
                row['test_loss'], row['test_acc'] = evaluate(model, testloader)
                row['train_loss_eval'], row['train_acc_eval'] = evaluate(model, train_eval_loader)
                message += (f' | train acc {row["train_acc_eval"]:.2f} | '
                            f'test loss {row["test_loss"]:.4f} acc {row["test_acc"]:.2f}')

            print(message)
            writer.writerow(row[k] for k in CSV_FIELDS)
            csv_file.flush()
            if diverged:
                break

    if diverged:
        (run_dir / 'DIVERGED').touch()
    else:
        torch.save(model.state_dict(), run_dir / 'model.pt')
        (run_dir / 'DONE').touch()
    print(f'{tag} finished in {int(time.time() - begin_time)} s')


def parse_args():
    p = argparse.ArgumentParser(description='Weight normalization experiments on CIFAR-10')
    p.add_argument('--methods', nargs='+', choices=list(METHODS), default=list(METHODS),
                   help='which methods to run (default: all)')
    p.add_argument('--seeds', nargs='+', type=int, default=[0], help='one run per method and seed')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=None, help='override the learning rate of every selected method')
    p.add_argument('--lr-decay', choices=['linear', 'none'], default='linear',
                   help='linear: constant lr, then linear decay to ~0 (default); none: constant lr')
    p.add_argument('--decay-start', type=int, default=None, help='first epoch of the decay (default: epochs // 2)')
    p.add_argument('--warmup-steps', type=int, default=0,
                   help='linear lr warm-up over the first N optimizer steps (one epoch = 500 steps); default: none')
    p.add_argument('--init-last-layer', choices=['data', 'plain'], default='data',
                   help="data-dependent init for the output layer too ('data', as in the paper) or plain WN init")
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=5, help='evaluate every N epochs (plus epoch 1 and the last)')
    p.add_argument('--no-augment', action='store_true', help='disable random translation + horizontal flip')
    p.add_argument('--noise', choices=['additive', 'relative'], default='additive',
                   help="input Gaussian noise: 'additive' (paper) or 'relative' (old behaviour of this repo)")
    p.add_argument('--init-mode', choices=['train', 'eval'], default='train',
                   help='noise/dropout on (train) or off (eval) while the init batch is propagated')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--data', default='./data')
    p.add_argument('--out', default='results')
    p.add_argument('--force', action='store_true', help='rerun runs that are already marked as done')
    return p.parse_args()


def main():
    args = parse_args()
    datasets = get_datasets(args.data, augment=not args.no_augment)
    for seed in args.seeds:
        for name in args.methods:
            run_experiment(name, METHODS[name], seed, args, datasets)


if __name__ == '__main__':
    main()
