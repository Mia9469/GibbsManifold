#!/usr/bin/env python3
"""
Held-out natural-image generalization test for Fig6 sparse coding models.

This is the most direct generalization test: dictionaries are learned on a
*subset* of the natural images and evaluated on patches from images the models
NEVER saw during training.  Unlike `generalization_test.py` (which evaluates the
paper's full-data checkpoints on synthetic 1/f noise and hand-crafted patterns),
this script trains an *isolated* set of models on an image-level train/test
split, so the comparison is a clean held-out generalization measurement.

Pipeline
--------
1. Split the 10 van-Hateren/Olshausen whitened images in IMAGES.mat into
   TRAIN (default 8) and HELD-OUT (default 2) by image index.
2. Train Null / SparseNet(Cauchy) / CVM-v2 on TRAIN patches only, with the exact
   training loop and hyper-parameters of train_new.py (n_runs independent seeds).
   Models are written to `record_heldout/` — the paper's `record/` is untouched.
3. Extract non-overlapping 16x16 patches from the HELD-OUT images.
4. Evaluate every learned dictionary on the held-out patches with the SAME
   standardized ISTA inference used in generalization_test.py (λ identical
   across dictionaries → isolates dictionary quality).
5. Report RE / SSIM / ICE / independence with effect sizes (Cohen's d) and
   bootstrap 95% CIs after aggregating seeds within each fold, so the held-out
   image is the unit of replication.

Usage
-----
  python heldout_generalization_test.py                 # train + test
  python heldout_generalization_test.py --skip-train    # reuse record_heldout/
  python heldout_generalization_test.py --n-runs 12 --n-holdout 2

Outputs → Fig6/generalization_results/held_out_natural/
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.model.SparseNetNew import SparseNet
from generalization_test import (evaluate_dictionary, add_stat_annotation)
from gen_stats import print_effect_table

DATA_PATH = os.path.join(BASE_DIR, 'src', 'model', 'IMAGES.mat')
RECORD_DIR = os.path.join(BASE_DIR, 'record_heldout')
OUTPUT_DIR = os.path.join(BASE_DIR, 'generalization_results', 'held_out_natural')

# (method key, train_new.py constraint name, dir suffix)
METHODS = [
    ('null',   'non',            'null'),
    ('cauchy', 's',              'cauchy'),
    ('cv_m',   'cv_constrained_v2', 'cv_m_v2'),
]

# Hyper-parameters matching the paper's main Fig6 experiment
PATCH_SIZE = 16
N_NEURONS = 144
N_PATCHES_PER_IMG = 1024     # matches train_new.py batch_size / NatPatchDataset N
LR_U = 1e-2
LR_R = 1e-2
LMDA = 5e-3
CV_RATIO = 0.40
ALPHA_S = 1.0
RECON_BUDGET = 0.60
DEFAULT_N_RUNS = 6
DEFAULT_N_EPOCHS = 10


# ═══════════════════════════════════════════════════════════════════════════
# Patch extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_train_patches(images, patch_size=PATCH_SIZE, n_per_img=N_PATCHES_PER_IMG):
    """
    Replicate NatPatchDataset's deterministic patch grid for a set of images,
    so the training distribution matches the paper exactly (just fewer images).

    images : ndarray (H, W, n_img)
    Returns: torch.Tensor (n_img * n_per_img, patch_size**2), mean-subtracted.
    """
    H = images.shape[0]
    n_img = images.shape[2]
    patches = torch.zeros((n_img * n_per_img, patch_size, patch_size))
    counter = 0
    for i in range(n_img):
        img = images[:, :, i]
        for j in range(n_per_img):
            x = j % int(n_per_img / patch_size) * patch_size % int(H)
            y = int(j / int(H / patch_size)) * patch_size
            crop = torch.tensor(img[x:x + patch_size, y:y + patch_size],
                                dtype=torch.float32)
            patches[counter] = crop - crop.mean()
            counter += 1
    return patches.reshape(patches.shape[0], -1)


def extract_test_patches(images, patch_size=PATCH_SIZE):
    """
    Clean non-overlapping grid of 16x16 patches from held-out images.
    Each patch mean-subtracted (consistent with training preprocessing).

    Returns: ndarray (n_patches, patch_size**2)
    """
    H, W = images.shape[0], images.shape[1]
    n_img = images.shape[2]
    ppr_r = H // patch_size
    ppr_c = W // patch_size
    out = []
    for i in range(n_img):
        img = images[:, :, i]
        for r in range(ppr_r):
            for c in range(ppr_c):
                patch = img[r * patch_size:(r + 1) * patch_size,
                            c * patch_size:(c + 1) * patch_size]
                out.append((patch - patch.mean()).ravel())
    return np.asarray(out, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Training (replicates train_new.py loop, restricted to TRAIN images)
# ═══════════════════════════════════════════════════════════════════════════

def fold_dir(fold, method):
    return os.path.join(RECORD_DIR, f'fold{fold}', f'trained_models_{method}')


def train_one(type_key, cons, model_key, train_patches, n_runs, n_epochs, device,
              fold, batch_size=1024):
    """Train n_runs independent models on train_patches; save final dict per run."""
    model_dir = fold_dir(fold, model_key)
    os.makedirs(model_dir, exist_ok=True)
    N = train_patches.shape[0]

    for run in range(n_runs):
        torch.manual_seed(1000 + 100 * fold + run)
        np.random.seed(1000 + 100 * fold + run)

        net = SparseNet(N_NEURONS, PATCH_SIZE, R_lr=LR_R, lmda=LMDA,
                        cons=cons, device=device, cv_ratio=CV_RATIO,
                        alpha_s=ALPHA_S, recon_budget=RECON_BUDGET)
        optim = torch.optim.SGD([{'params': net.U.weight, 'lr': LR_U}])

        for e in range(n_epochs):
            perm = torch.randperm(N)
            for b0 in range(0, N, batch_size):
                idx = perm[b0:b0 + batch_size]
                img_batch = train_patches[idx].to(device)
                pred = net(img_batch)                       # ISTA inner loop
                loss = net.get_total_loss(img_batch, pred)  # recon + constraint
                optim.zero_grad()
                loss.backward()
                optim.step()
                net.normalize_weights()

        torch.save(net, os.path.join(model_dir, f'ckpt-{run}.pth'))


def models_exist(folds, n_runs):
    for fold in folds:
        for _, _, model_key in METHODS:
            if not os.path.exists(os.path.join(fold_dir(fold, model_key),
                                               f'ckpt-{n_runs - 1}.pth')):
                return False
    return True


def method_models_exist(folds, model_key, n_runs):
    for fold in folds:
        if not os.path.exists(os.path.join(fold_dir(fold, model_key),
                                           f'ckpt-{n_runs - 1}.pth')):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_method(type_key, model_key, test_patches, n_runs, fold, device):
    """Evaluate every per-run dictionary (this fold) on the held-out patches."""
    model_dir = fold_dir(fold, model_key)
    patch_size = int(np.sqrt(test_patches.shape[1]))
    results = []
    for run in range(n_runs):
        ckpt = os.path.join(model_dir, f'ckpt-{run}.pth')
        if not os.path.exists(ckpt):
            continue
        net = torch.load(ckpt, map_location='cpu', weights_only=False)
        U = net.U.weight.detach().cpu().numpy()       # [M^2, K]
        try:
            res = evaluate_dictionary(U, test_patches, patch_size, device)
        except FloatingPointError as exc:
            print(f"  [{type_key}] fold={fold} run={run}: skipped ({exc})")
            continue
        res['type'] = type_key
        res['run'] = run
        res['fold'] = fold
        results.append(res)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_panels(df, output_dir):
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['axes.labelsize'] = 13
    palette = {'Null': '#868686FF', 'SparseNet': '#0073C2FF', 'CVM': '#EFC000FF'}
    label_map = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
    order = ['Null', 'SparseNet', 'CVM']
    box_pairs = [('null', 'cauchy'), ('null', 'cv_m'), ('cauchy', 'cv_m')]

    pdf = df.copy()
    pdf['label'] = pdf['type'].map(label_map)
    features = ['RE', 'SSIM']
    ylabels = {'RE': 'Held-out RE  (lower = better)',
               'SSIM': 'Held-out SSIM  (higher = better)'}

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150, facecolor='w')
    for ax, feat in zip(axes, features):
        sns.boxplot(x='label', y=feat, data=pdf, order=order, hue='label',
                    palette=palette, legend=False, width=0.6, ax=ax)
        sns.stripplot(x='label', y=feat, data=pdf, order=order, color='0.25',
                      size=3, alpha=0.6, ax=ax)
        add_stat_annotation(ax, df, 'type', feat, box_pairs,
                            ['null', 'cauchy', 'cv_m'], paired_unit='fold')
        ax.set_ylabel(ylabels[feat]); ax.set_xlabel('')
    fig.suptitle('Held-out natural images (leave-one-image-out CV)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(output_dir, 'heldout_panels.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _json_safe(rows):
    return [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
             for k, v in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--n-runs', type=int, default=DEFAULT_N_RUNS,
                    help='Independent seeds per fold (default %(default)s)')
    ap.add_argument('--n-epochs', type=int, default=DEFAULT_N_EPOCHS)
    ap.add_argument('--cv', choices=['loo', 'split'], default='loo',
                    help="'loo' = leave-one-image-out CV over all images "
                         "(default); 'split' = single train/test split")
    ap.add_argument('--n-holdout', type=int, default=2,
                    help="For --cv split: images held out (default 2)")
    args = ap.parse_args()

    device = torch.device('cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X = loadmat(DATA_PATH)['IMAGES']           # (512, 512, 10)
    n_img = X.shape[2]

    # ── Define folds: each fold = (held-out test indices, train indices) ──
    if args.cv == 'loo':
        folds = [([i], [j for j in range(n_img) if j != i]) for i in range(n_img)]
        print(f"Leave-one-image-out CV: {n_img} folds, "
              f"{args.n_runs} seeds/fold → {n_img * args.n_runs} models/method")
    else:
        nh = args.n_holdout
        folds = [(list(range(n_img - nh, n_img)), list(range(n_img - nh)))]
        print(f"Single split: train {folds[0][1]}, held-out {folds[0][0]}")

    fold_ids = list(range(len(folds)))

    # ── Train ──
    if not args.skip_train and not models_exist(fold_ids, args.n_runs):
        print("\nTraining isolated CV models (record_heldout/)...")
        t0 = time.time()
        for fi, (test_idx, train_idx) in enumerate(folds):
            train_patches = extract_train_patches(X[:, :, train_idx])
            print(f"\n── Fold {fi} (held-out img {test_idx}) | "
                  f"train patches {tuple(train_patches.shape)} ──")
            for type_key, cons, model_key in METHODS:
                if method_models_exist([fi], model_key, args.n_runs):
                    print(f"   {type_key} already present; skipping")
                    continue
                train_one(type_key, cons, model_key, train_patches,
                          args.n_runs, args.n_epochs, device, fold=fi)
                print(f"   trained {type_key}")
        print(f"\nTraining done in {time.time() - t0:.0f}s")
    elif args.skip_train and not models_exist(fold_ids, args.n_runs):
        print("[WARN] --skip-train but models missing in record_heldout/.")
    else:
        print("\nModels already present — skipping training.")

    # ── Evaluate every fold on its held-out image ──
    print(f"\n{'─' * 70}\n  Evaluating held-out images (per fold)\n{'─' * 70}")
    per_run = []
    for fi, (test_idx, _) in enumerate(folds):
        test_patches = extract_test_patches(X[:, :, test_idx])
        for type_key, _, model_key in METHODS:
            res = evaluate_method(type_key, model_key, test_patches,
                                  args.n_runs, fi, device)
            per_run.extend(res)
        msg = "  ".join(
            f"{m}:SSIM={np.mean([r['SSIM'] for r in per_run if r['type']==m and r['fold']==fi]):.3f}"
            for m in ['null', 'cauchy', 'cv_m'])
        print(f"  fold {fi} (img {test_idx}): {msg}")

    if not per_run:
        print("[ERROR] No results — did training run?")
        return

    # ── Aggregate to FOLD level (image = unit of replication) ──
    df_run = pd.DataFrame(per_run)
    metrics = ['RE', 'SSIM', 'ICE', 'mean_a', 'independence']
    fold_rows = (df_run.groupby(['type', 'fold'])[metrics]
                 .mean().reset_index().to_dict('records'))

    # ── Report (fold-level) ──
    label_map = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
    unit = (f'Held-out natural images — leave-one-image-out CV '
            f'(n={len(folds)} image-folds)') if args.cv == 'loo' \
        else 'Held-out natural images (single split)'
    print_effect_table(fold_rows, metrics, ['null', 'cauchy', 'cv_m'], label_map,
                       [('null', 'cauchy'), ('null', 'cv_m'), ('cauchy', 'cv_m')],
                       unit, unit_label='held-out image fold', paired_unit='fold')

    # ── Plot + save (fold-level summary feeds the composite panel) ──
    plot_panels(pd.DataFrame(fold_rows), OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(_json_safe(fold_rows), f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'summary_per_run.json'), 'w') as f:
        json.dump(_json_safe(per_run), f, indent=2)
    print(f"  Saved → {os.path.join(OUTPUT_DIR, 'summary.json')} "
          f"(fold-level, n={len(folds)}) + summary_per_run.json")


if __name__ == '__main__':
    main()
