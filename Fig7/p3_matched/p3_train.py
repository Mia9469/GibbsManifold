"""
P3 matched-baseline training driver.

Runs a single (method, knob, seed) configuration end-to-end:
  - reproducible random init via --seed
  - shared training data, optimiser, ISTA settings across all three methods
  - records every per-image metric needed for the post-hoc matched-RE / matched-a
    Pareto analysis (RE, mean |a|, population CV, lifetime CV, decorrelation)
  - writes one JSON record per run plus the final torch checkpoint

Designed to be called from a server batch script (p3_sweep.sh) once per
(method, knob, seed).  No global state; can run in parallel.

Usage examples
--------------
  # Null baseline at L1 = 5e-3, seed 0
  python p3_train.py --method null --lmda 5e-3 --seed 0

  # Cauchy with alpha_s = 1.0
  python p3_train.py --method cauchy --lmda 5e-3 --alpha_s 1.0 --seed 0

  # CV-max with cv_ratio = 0.10
  python p3_train.py --method cvmax --lmda 5e-3 --cv_ratio 0.10 --seed 0

Outputs
-------
  results/records/<run_id>.json     -- scalar metrics per epoch + final eval
  results/checkpoints/<run_id>.pth  -- trained model state_dict
  results/log/<run_id>.log          -- console log

where <run_id> = "{method}__lmda{lmda}_a{alpha_s}_cv{cv_ratio}__seed{seed}"
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# -- locate project root and import shared model code ------------------------
HERE = Path(__file__).resolve().parent
FIG7 = HERE.parent
ROOT = FIG7.parent
sys.path.insert(0, str(FIG7))

from src.model.SparseNetNew import SparseNet                     # noqa: E402
from src.model.ImageDataset1 import NatPatchDataset              # noqa: E402

DATA_PATH = FIG7 / 'src' / 'model' / 'IMAGES.mat'
RESULTS = HERE / 'results'
RECORDS_DIR = RESULTS / 'records'
CHKPT_DIR = RESULTS / 'checkpoints'
LOG_DIR = RESULTS / 'log'
for d in (RECORDS_DIR, CHKPT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='P3 matched-baseline trainer')
    # Method selector
    p.add_argument('--method', required=True,
                   choices=['null', 'cauchy', 'cvmax', 'cvmax_v2'],
                   help="null = L1-only, cauchy = L1+Cauchy prior, "
                        "cvmax = L1+CV maximisation (cv_constrained: ISTA gated, "
                        "U-update unconditional), "
                        "cvmax_v2 = ISTA gated AND U-update gated (U only learns "
                        "CV when last ISTA iter had CV active)")
    # Common knobs
    p.add_argument('--lmda', type=float, default=5e-3,
                   help='L1 soft-thresholding strength (always active)')
    p.add_argument('--alpha_s', type=float, default=0.0,
                   help='Cauchy weight (only used when method=cauchy)')
    p.add_argument('--cv_ratio', type=float, default=0.0,
                   help='CV penalty / recon target ratio '
                        '(only used when method=cvmax)')
    p.add_argument('--recon_budget', type=float, default=1.0,
                   help='cvmax (v1, cv_constrained): warmup-relative gate. '
                        'CV penalty fires only when recon ≤ recon_budget × '
                        'recon_warmup. 1.0=strict; >1.0 allows recon drift. '
                        'cvmax_v2 (cv_constrained_v2): absolute RE target. '
                        'CV only fires when current RE ≤ recon_budget. '
                        'Use values <1.0 (e.g. 0.4-0.6) to force the gate to '
                        'actually engage.')
    # Architecture
    p.add_argument('-K', '--n_neuron', type=int, default=144)
    p.add_argument('-M', '--patch_size', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=1024)
    # Training
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--r_lr', type=float, default=1e-2)
    # Reproducibility
    p.add_argument('--seed', type=int, default=0)
    # I/O
    p.add_argument('--data_path', type=str, default=str(DATA_PATH))
    p.add_argument('--results_dir', type=str, default=str(RESULTS))
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_id(args: argparse.Namespace) -> str:
    """Build a deterministic run id.

    Includes recon_budget so 2-D cvmax sweeps (recon_budget × cv_ratio) do
    not collide.  null and cauchy will always have b=1.0, which is
    harmless and keeps the format uniform.
    """
    return (f"{args.method}__lmda{args.lmda:g}"
            f"_a{args.alpha_s:g}_cv{args.cv_ratio:g}_b{args.recon_budget:g}"
            f"__seed{args.seed}")


def method_to_cons(method: str) -> str:
    """Map P3 method label → SparseNet cons argument.

    cvmax    → cv_constrained    (ISTA gated, U-update unconditional)
    cvmax_v2 → cv_constrained_v2 (ISTA gated AND U-update gated).
              v2 lets U temporarily learn pure-recon when ISTA's gate
              closed, so the dictionary can recover RE before being
              pulled back toward CV.  Hypothesis: v2 unlocks lower RE
              than v1 by removing the unconditional CV pull on U.
    """
    return {
        'null': 'non',
        'cauchy': 's',
        'cvmax': 'cv_constrained',
        'cvmax_v2': 'cv_constrained_v2',
    }[method]


def evaluate_model(net: SparseNet, dataloader: DataLoader,
                   device: torch.device) -> dict:
    """Evaluate on (deterministic) batches and return metric dict.

    All metrics are averaged across batches, weighted equally.  These are
    the quantities consumed by p3_eval.py for matched-RE / matched-a
    analysis.
    """
    net.eval()
    re_acc = mean_a_acc = ent_acc = 0.0
    cv_pop_acc = cv_lt_acc = 0.0
    n_batches = 0

    # NOTE: net.ista_() does *internal* backward passes on net.R, so we cannot
    # wrap this loop in torch.no_grad().  Instead we run ISTA with grads
    # enabled (they only flow into net.R, which is local to ista_), then
    # detach R and freeze U.weight for the metric computation so no graph is
    # built on top of the ISTA result.
    u_was_requires_grad = net.U.weight.requires_grad
    net.U.weight.requires_grad_(False)
    try:
        for img_batch in dataloader:
            img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)
            # Forward (re-runs ISTA; needs grads on R internally)
            net.ista_(img_batch)
            with torch.no_grad():
                R = net.R.detach()
                R_abs = torch.abs(R)
                pred = net.U(R)

                # RE per batch = ||img - pred||_2 / ||img||_2
                re = torch.norm(img_batch - pred) / (torch.norm(img_batch) + 1e-12)
                re_acc += re.item()

                mean_a_acc += R_abs.mean().item()

                # population CV (per-patch then averaged)
                mu = R_abs.mean(dim=1)
                sig = R_abs.std(dim=1)
                cv_pop_acc += (sig / (mu + 1e-8)).mean().item()
                # lifetime CV (per-neuron then averaged)
                mu_n = R_abs.mean(dim=0)
                sig_n = R_abs.std(dim=0)
                cv_lt_acc += (sig_n / (mu_n + 1e-8)).mean().item()

                # entropy per neuron (200-bin histogram on |a|, averaged across K)
                R_np = R_abs.cpu().numpy()
                K = R_np.shape[1]
                ent = 0.0
                for v in range(K):
                    hist, _ = np.histogram(R_np[:, v], bins=200, range=(0, 2))
                    p = hist / max(hist.sum(), 1)
                    p = p[p > 0]
                    ent += float(-np.sum(p * np.log2(p)))
                ent_acc += ent / K

                n_batches += 1
    finally:
        net.U.weight.requires_grad_(u_was_requires_grad)

    n = max(n_batches, 1)
    re_avg = re_acc / n
    mean_a = mean_a_acc / n
    return {
        'RE': re_avg,
        'mean_a': mean_a,
        'CV_pop': cv_pop_acc / n,
        'CV_lt': cv_lt_acc / n,
        'entropy_per_neuron': ent_acc / n,
        # ICE definitions (both reported; eval script picks one)
        'ICE_log_a': float(-np.log2(re_avg + 1e-12) / (mean_a + 1e-12)),
        'ICE_ent_a': float((ent_acc / n) / (mean_a + 1e-12)),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    rid = run_id(args)

    # Tee stdout into a log file alongside the record
    log_path = LOG_DIR / f'{rid}.log'
    rec_path = RECORDS_DIR / f'{rid}.json'
    ckpt_path = CHKPT_DIR / f'{rid}.pth'

    log_fh = open(log_path, 'w', buffering=1)
    def log(msg):
        print(msg)
        log_fh.write(msg + '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'[p3_train] run_id = {rid}')
    log(f'[p3_train] device = {device}')
    log(f'[p3_train] args = {vars(args)}')

    # Build model with method-specific knobs
    cons = method_to_cons(args.method)
    net = SparseNet(
        K=args.n_neuron, M=args.patch_size,
        R_lr=args.r_lr, lmda=args.lmda,
        cons=cons, device=device,
        cv_ratio=args.cv_ratio,
        alpha_s=args.alpha_s,
        recon_budget=args.recon_budget,
    )

    # Data
    dataset = NatPatchDataset(args.batch_size, args.patch_size,
                              args.patch_size, fpath=args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Optimiser for U
    optim = torch.optim.SGD([{'params': net.U.weight, 'lr': args.lr}])

    # ── Train loop ──────────────────────────────────────────────────────
    epoch_log = []
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        running = {'loss': 0.0, 'RE_train': 0.0, 'mean_a': 0.0,
                   'CV_pop': 0.0, 'CV_lt': 0.0}
        n_batches = 0
        for img_batch in dataloader:
            img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)
            pred = net(img_batch)
            loss = net.get_total_loss(img_batch, pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
            net.normalize_weights()

            with torch.no_grad():
                running['loss'] += loss.item()
                # RE_train uses sum-squared (E in SparseNet) divided by ||img||
                running['RE_train'] += float(
                    torch.norm(img_batch - pred) /
                    (torch.norm(img_batch) + 1e-12)
                )
                running['mean_a'] += float(torch.abs(net.R).mean())
                running['CV_pop'] += net.cv_val
                running['CV_lt'] += net.cv_lifetime
            n_batches += 1

        for k in running:
            running[k] /= max(n_batches, 1)
        running['epoch'] = ep + 1
        epoch_log.append(running)
        log(f'[ep {ep+1:02d}/{args.epochs}] loss={running["loss"]:.2f} '
            f'RE_train={running["RE_train"]:.4f} '
            f'mean_a={running["mean_a"]:.4f} '
            f'CV_pop={running["CV_pop"]:.3f}')

    train_time = time.time() - t0
    log(f'[p3_train] training done in {train_time:.1f}s')

    # ── Final evaluation ────────────────────────────────────────────────
    eval_metrics = evaluate_model(net, dataloader, device)
    log(f'[p3_train] eval: {json.dumps(eval_metrics, indent=2)}')

    # ── Save checkpoint + record ────────────────────────────────────────
    torch.save({
        'state_dict': net.state_dict(),
        'config': vars(args),
        'cons': cons,
    }, ckpt_path)
    log(f'[p3_train] saved checkpoint -> {ckpt_path}')

    record = {
        'run_id': rid,
        'method': args.method,
        'cons': cons,
        'config': vars(args),
        'train_time_sec': train_time,
        'epoch_log': epoch_log,
        'eval': eval_metrics,
    }
    with open(rec_path, 'w') as f:
        json.dump(record, f, indent=2)
    log(f'[p3_train] saved record    -> {rec_path}')

    log_fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
