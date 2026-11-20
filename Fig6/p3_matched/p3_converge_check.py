"""
P3 convergence diagnostic.

Question: is 10 epochs enough for the matched-baseline sweep, or does RE keep
falling well past epoch 10?  This script runs ONE representative configuration
per method for 50 epochs (single seed) and records per-epoch evaluation
metrics on a deterministic eval pass.

It is intentionally minimal -- just enough to look at the four headline
quantities (RE, mean|a|, CV_pop, ICE_log_a) vs. epoch.

Representative knob values picked to land near the centre of each method's
sweep range (i.e. neither degenerate nor saturated):
    null    : lmda     = 5e-3   (the matched baseline used for cauchy/cvmax)
    cauchy  : alpha_s  = 1.0
    cvmax   : cv_ratio = 0.10

Outputs
-------
    results/converge/converge.csv    -- one row per (method, epoch)
    results/converge/converge.pdf    -- 4-panel: RE, mean_a, CV_pop, RAE vs ep
    results/converge/<method>_50ep.json  -- full per-config record

Usage
-----
    python p3_converge_check.py                # 3 configs x 50 epochs
    python p3_converge_check.py --epochs 30
    python p3_converge_check.py --methods cvmax
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG7 = HERE.parent
sys.path.insert(0, str(FIG7))

from src.model.SparseNetNew import SparseNet                # noqa: E402
from src.model.ImageDataset1 import NatPatchDataset         # noqa: E402

DATA_PATH = FIG7 / 'src' / 'model' / 'IMAGES.mat'
OUT_DIR = HERE / 'results' / 'converge'

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9

CONFIGS = {
    'null':   dict(cons='non', lmda=5e-3, alpha_s=0.0, cv_ratio=0.0),
    'cauchy': dict(cons='s',   lmda=5e-3, alpha_s=1.0, cv_ratio=0.0),
    'cvmax':  dict(cons='cv',  lmda=5e-3, alpha_s=0.0, cv_ratio=0.10),
}
COLORS = {'null': '0.4', 'cauchy': 'C0', 'cvmax': 'C3'}
LABELS = {'null': 'null  (lmda=5e-3)',
          'cauchy': 'cauchy (alpha_s=1.0)',
          'cvmax':  'cvmax  (cv_ratio=0.10)'}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='P3 convergence diagnostic')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-K', '--n_neuron', type=int, default=144)
    p.add_argument('-M', '--patch_size', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--r_lr', type=float, default=1e-2)
    p.add_argument('--methods', type=str, default='null,cauchy,cvmax',
                   help='Comma-separated subset of {null,cauchy,cvmax}')
    p.add_argument('--data_path', type=str, default=str(DATA_PATH))
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_metrics(net: SparseNet, dataloader: DataLoader,
                 device: torch.device) -> dict:
    """Deterministic eval pass; returns aggregate scalar metrics.

    Mirrors p3_train.evaluate_model: ista_ runs its own internal backward
    on R, so we cannot use torch.no_grad() around the loop.  Instead we
    freeze U.weight (no grads accumulate on it during ista_), then read
    metrics under no_grad, then restore the original requires_grad flag.
    """
    net.eval()
    re_acc = mean_a_acc = 0.0
    cv_pop_acc = cv_lt_acc = 0.0
    n = 0
    u_was_requires_grad = net.U.weight.requires_grad
    net.U.weight.requires_grad_(False)
    try:
        for img_batch in dataloader:
            img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)
            net.ista_(img_batch)
            with torch.no_grad():
                R = net.R.detach()
                R_abs = torch.abs(R)
                pred = net.U(R)
                re = torch.norm(img_batch - pred) / (torch.norm(img_batch) + 1e-12)
                re_acc += re.item()
                mean_a_acc += R_abs.mean().item()
                mu = R_abs.mean(dim=1); sig = R_abs.std(dim=1)
                cv_pop_acc += (sig / (mu + 1e-8)).mean().item()
                mu_n = R_abs.mean(dim=0); sig_n = R_abs.std(dim=0)
                cv_lt_acc += (sig_n / (mu_n + 1e-8)).mean().item()
            n += 1
    finally:
        net.U.weight.requires_grad_(u_was_requires_grad)
    n = max(n, 1)
    re = re_acc / n
    ma = mean_a_acc / n
    return {
        'RE': re,
        'mean_a': ma,
        'CV_pop': cv_pop_acc / n,
        'CV_lt': cv_lt_acc / n,
        'ICE_log_a': float(-np.log2(re + 1e-12) / (ma + 1e-12)),
    }


def train_one(method: str, args: argparse.Namespace) -> list[dict]:
    """Train one config for args.epochs epochs, eval after every epoch.

    Returns a list of dicts (one per epoch) with all eval metrics.
    """
    cfg = CONFIGS[method]
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SparseNet(
        K=args.n_neuron, M=args.patch_size,
        R_lr=args.r_lr, lmda=cfg['lmda'],
        cons=cfg['cons'], device=device,
        cv_ratio=cfg['cv_ratio'],
        alpha_s=cfg['alpha_s'],
    )
    dataset = NatPatchDataset(args.batch_size, args.patch_size,
                              args.patch_size, fpath=args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    optim = torch.optim.SGD([{'params': net.U.weight, 'lr': args.lr}])

    log: list[dict] = []
    print(f'[converge] method={method}  cfg={cfg}  device={device}')
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        for img_batch in dataloader:
            img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)
            pred = net(img_batch)
            loss = net.get_total_loss(img_batch, pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
            net.normalize_weights()
        m = eval_metrics(net, dataloader, device)
        m['epoch'] = ep
        m['method'] = method
        log.append(m)
        print(f'  [{method:6s} ep {ep:02d}/{args.epochs}] '
              f'RE={m["RE"]:.4f}  mean_a={m["mean_a"]:.4f}  '
              f'CV_pop={m["CV_pop"]:.3f}  ICE={m["ICE_log_a"]:.2f}')
    print(f'[converge] {method} done in {time.time()-t0:.1f}s')
    return log


def write_csv(path: Path, all_logs: list[dict]) -> None:
    if not all_logs:
        return
    keys = ['method', 'epoch', 'RE', 'mean_a',
            'CV_pop', 'CV_lt', 'ICE_log_a']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_logs:
            w.writerow({k: r[k] for k in keys})


def plot(all_logs: list[dict], out_path: Path, n_epochs: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    panels = [
        ('RE',         'Reconstruction error'),
        ('mean_a',     r'mean $|a|$'),
        ('CV_pop',     'Population CV'),
        ('ICE_log_a',  'ICE  (bits / activation)'),
    ]
    for ax, (key, ylab) in zip(axes.ravel(), panels):
        for method in CONFIGS:
            sub = [r for r in all_logs if r['method'] == method]
            if not sub:
                continue
            x = np.array([r['epoch'] for r in sub])
            y = np.array([r[key]    for r in sub])
            ax.plot(x, y, '-', lw=1.6, color=COLORS[method],
                    label=LABELS[method])
            ax.plot(10, np.interp(10, x, y), 'o', ms=6,
                    mec=COLORS[method], mfc='white', mew=1.5)
        ax.axvline(10, ls=':', color='0.5', lw=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8, loc='best')
    fig.suptitle(f'Convergence diagnostic over {n_epochs} epochs '
                 r'(dotted line = current main experiment cutoff)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    bad = [m for m in methods if m not in CONFIGS]
    if bad:
        print(f'[converge] unknown method(s): {bad}', file=sys.stderr)
        return 1

    all_logs: list[dict] = []
    for method in methods:
        log = train_one(method, args)
        all_logs.extend(log)
        with open(OUT_DIR / f'{method}_{args.epochs}ep.json', 'w') as f:
            json.dump({'method': method,
                       'config': CONFIGS[method],
                       'args': vars(args),
                       'epoch_log': log}, f, indent=2)

    write_csv(OUT_DIR / 'converge.csv', all_logs)
    plot(all_logs, OUT_DIR / 'converge.pdf', args.epochs)
    print(f'\n[converge] wrote {OUT_DIR/"converge.csv"}')
    print(f'[converge] wrote {OUT_DIR/"converge.pdf"}')

    # Console summary: by what epoch has each curve plateaued?
    print('\n=== plateau diagnostic ===')
    print(f'(comparing epoch 10 vs epoch {args.epochs})')
    for method in methods:
        sub = [r for r in all_logs if r['method'] == method]
        if not sub:
            continue
        e10 = next((r for r in sub if r['epoch'] == 10), None)
        eN  = sub[-1]
        if e10 is None:
            continue
        print(f'\n  {method}:')
        for k in ['RE', 'mean_a', 'CV_pop', 'ICE_log_a']:
            d_pct = 100.0 * (eN[k] - e10[k]) / max(abs(e10[k]), 1e-12)
            print(f'    {k:>10s}: ep10={e10[k]:8.4f}  '
                  f'ep{args.epochs}={eN[k]:8.4f}  '
                  f'(Δ = {d_pct:+.1f}%)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
