"""
Unified training script for SparseNet with three constraint types.

Fixes vs original train.py/train_s.py/train_cv.py:
  1. U update includes constraint penalty (not just reconstruction)
  2. Cauchy sign corrected: +sparsity_term (penalise, not reward large R)
  3. CV computed per-neuron then averaged (not global CV of entire R matrix)
  4. All three types use identical data (no noise confound)
  5. Single script, controlled by --cons argument

Usage:
  python train_new.py --cons non   # null baseline
  python train_new.py --cons s     # Cauchy sparsity
  python train_new.py --cons cv    # CV maximisation
"""

import os
import sys
import argparse
import time
import random

sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.model.SparseNetNew import SparseNet
from src.model.ImageDataset1 import NatPatchDataset
from src.scripts.plotting import plot_rf


def parse_args():
    p = argparse.ArgumentParser(description="Train SparseNet (fixed)")
    p.add_argument('--cons', type=str, default='non',
                   choices=['non', 's', 'cv',
                            'cv_constrained', 'cv_constrained_v2'],
                   help="Constraint: non (null), s (Cauchy), cv (CV-max), "
                        "cv_constrained (v1: warmup-relative gate), "
                        "cv_constrained_v2 (v2: absolute RE-target gate — "
                        "recon_budget is the target RE, e.g. 0.6)")
    p.add_argument('-K', '--n_neuron', type=int, default=144)
    p.add_argument('-M', '--size', type=int, default=16)
    p.add_argument('-N', '--batch_size', type=int, default=1024)
    p.add_argument('-e', '--epoch', type=int, default=100)
    p.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    p.add_argument('-rlr', '--r_learning_rate', type=float, default=1e-2)
    p.add_argument('-lmda', '--reg', type=float, default=5e-3)
    p.add_argument('--cv_ratio', type=float, default=0.1,
                   help="Target |CV_penalty|/|recon| ratio (adaptive scaling)")
    p.add_argument('--alpha_s', type=float, default=1.0,
                   help="Weight for Cauchy sparsity constraint")
    p.add_argument('--recon-budget', type=float, default=1.0,
                   help="cv_constrained (v1): gate threshold = recon_budget × "
                        "warmup_recon (1.0 = strict; 1.2 = +20%% recon drift). "
                        "cv_constrained_v2: absolute target RE — CV only fires "
                        "when current RE ≤ recon_budget (use 0.55-0.60).")
    p.add_argument('--n_runs', type=int, default=20,
                   help="Number of independent runs (random seeds)")
    p.add_argument('--run_start', type=int, default=0,
                   help="Starting run index")
    p.add_argument('--seed-base', type=int, default=20260630,
                   help="Run n uses seed_base + n (default: %(default)s)")
    p.add_argument('--data_path', type=str,
                   default='./src/model/IMAGES.mat',
                   help="Path to IMAGES.mat")
    p.add_argument('--output_dir', type=str, default='./record',
                   help="Base output directory")
    p.add_argument('--tag', type=str, default='',
                   help="Optional tag appended to directory names (e.g. 'cv0.04')")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Seed every RNG used by model initialization and patch sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    arg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Constraint: {arg.cons}")

    # Naming convention matching original: _16 (null), _cauchy_16_nl, _cv_16_nl
    name_map = {
        'non':               '_16_new',
        's':                 '_cauchy_16_nl_new',
        'cv':                '_cv_16_nl_new',
        'cv_constrained':    '_cv_con_new',
        'cv_constrained_v2': '_cv_v2_new',
    }
    suffix = name_map[arg.cons]
    if arg.tag:
        suffix = suffix + '_' + arg.tag

    model_dir = os.path.join(arg.output_dir, f'trained_models{suffix}')
    os.makedirs(model_dir, exist_ok=True)

    for n in range(arg.run_start, arg.run_start + arg.n_runs):
        run_seed = arg.seed_base + n
        set_seed(run_seed)
        print(f"\n{'='*60}")
        print(f"Run {n} | seed={run_seed} | constraint={arg.cons}")
        print(f"{'='*60}")

        # Tensorboard
        board = SummaryWriter(
            os.path.join(arg.output_dir, f"runs{suffix}", f"sparsenet_{n}")
        )

        # Create model
        sparse_net = SparseNet(
            arg.n_neuron, arg.size,
            R_lr=arg.r_learning_rate, lmda=arg.reg,
            cons=arg.cons, device=device,
            cv_ratio=arg.cv_ratio, alpha_s=arg.alpha_s,
            recon_budget=arg.recon_budget,
        )

        # Load data (deterministic patches, no noise)
        dataloader = DataLoader(
            NatPatchDataset(arg.batch_size, arg.size, arg.size,
                            fpath=arg.data_path),
            batch_size=1024
        )

        # Optimizer for U only
        optim = torch.optim.SGD(
            [{'params': sparse_net.U.weight, 'lr': arg.learning_rate}]
        )

        # Metric accumulators
        record_loss = []
        record_CV = []
        record_CV_lt = []   # lifetime CV (supplementary)
        record_S = []
        record_E = []
        record_sigma = []
        record_mu = []

        time1 = time.strftime("%H:%M:%S")

        for e in range(arg.epoch):
            running = {'loss': 0, 'E': 0, 'CV': 0, 'CV_lt': 0, 'S': 0, 'm': 0, 'sig': 0}
            c = 0

            for img_batch in tqdm(dataloader,
                                  desc=f'Run {n} Epoch {e+1}/{arg.epoch}',
                                  total=len(dataloader)):
                img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)

                # Forward: ISTA finds R, then pred = U(R)
                pred = sparse_net(img_batch)

                # *** KEY FIX: U loss includes constraint ***
                loss = sparse_net.get_total_loss(img_batch, pred)

                running['loss'] += loss.item()
                running['E'] += sparse_net.E
                running['CV'] += sparse_net.cv_val
                running['CV_lt'] += sparse_net.cv_lifetime
                running['S'] += sparse_net.S
                running['m'] += sparse_net.m
                running['sig'] += sparse_net.sig

                # Update U
                optim.zero_grad()
                loss.backward()
                optim.step()

                # Normalise dictionary columns
                sparse_net.normalize_weights()

                c += 1

                # Save checkpoint at last epoch
                if (e + 1) == arg.epoch:
                    pth_path = os.path.join(model_dir,
                                            f'ckpt-{e+1}-{n}-{c}.pth')
                    torch.save(sparse_net, pth_path)

            # Epoch averages
            record_loss.append(running['loss'] / c)
            record_E.append(running['E'] / c)
            record_CV.append(running['CV'] / c)
            record_CV_lt.append(running['CV_lt'] / c)
            record_S.append(running['S'] / c)
            record_mu.append(running['m'] / c)
            record_sigma.append(running['sig'] / c)

            board.add_scalar('Loss', running['loss'] / c, e)
            board.add_scalar('E', running['E'] / c, e)
            board.add_scalar('CV_pop', running['CV'] / c, e)
            board.add_scalar('CV_life', running['CV_lt'] / c, e)
            board.add_scalar('S', running['S'] / c, e)

            if (e + 1) % 10 == 0:
                print(f"  Epoch {e+1}: loss={running['loss']/c:.1f}  "
                      f"E={running['E']/c:.1f}  CV={running['CV']/c:.4f}  "
                      f"S={running['S']/c:.1f}")

        time2 = time.strftime("%H:%M:%S")
        print(f"Run {n} done: {time1} → {time2}")

        # Save training record
        rec_path = os.path.join(arg.output_dir,
                                f'record{suffix}_{n}.mat')
        scio.savemat(rec_path, {
            "CV": record_CV,
            "CV_lt": record_CV_lt,
            "E": record_E,
            "loss": record_loss,
            "mu": record_mu,
            "S": record_S,
            "sigma": record_sigma,
            "time": [time2, time1],
            "seed": run_seed,
            "run_index": n,
            "constraint": arg.cons,
            "cv_ratio": arg.cv_ratio,
            "alpha_s": arg.alpha_s,
            "recon_budget": arg.recon_budget,
            "lmda": arg.reg,
            "n_epochs": arg.epoch,
        })
        print(f"Saved record → {rec_path}")

        board.close()


if __name__ == '__main__':
    main()
