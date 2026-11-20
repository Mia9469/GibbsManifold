#!/usr/bin/env python3
"""
Generalization test for Fig6 sparse coding models.

Tests how well dictionaries learned by Null, SparseNet (Cauchy), and CVM
generalize to new images NOT seen during training.

Two test sets:
  1. 1/f (pink noise) images — synthetic images with natural-image-like
     second-order statistics but wholly out-of-distribution.
  2. Structured patterns — oriented gratings, edges, and bars that test
     whether CVM's Gabor-like basis functions represent oriented structure
     better than the alternatives.

For each model's dictionary, we run a standardized ISTA inference (same
λ = 5e-3 across all dictionaries) to isolate dictionary quality from
inference-hyperparameter confounding.  CVM uses the manuscript's final
absolute-RE-gated v2 algorithm (`cv_constrained_v2`).

Usage:
  python generalization_test.py                  # train + test (full)
  python generalization_test.py --skip-train     # use existing checkpoints
  python generalization_test.py --n-runs 5 --seed 42

Outputs → Fig6/generalization_results/
"""

import os
import sys
import argparse
import subprocess
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import structural_similarity as ssim
from scipy.io import loadmat
import seaborn as sns
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.model.SparseNetNew import SparseNet
from src.model.ImageDataset1 import NatPatchDataset
from basis_analysis import decorrelation
from gen_stats import cohens_d, d_magnitude, print_effect_table, paired_values
from evaluation_metrics import reference_ssim, exact_unique_inputs, validate_frozen_manifest

DATA_PATH = os.path.join(BASE_DIR, 'src', 'model', 'IMAGES.mat')
RECORD_DIR = os.path.join(BASE_DIR, 'record')
OUTPUT_DIR = os.path.join(BASE_DIR, 'generalization_results')

# Model directory naming (matching train_new.py)
MODEL_DIRS = {
    'null':   os.path.join(RECORD_DIR, 'trained_models_16_new'),
    'cauchy': os.path.join(RECORD_DIR, 'trained_models_cauchy_16_nl_new'),
    'cv_m':   os.path.join(RECORD_DIR, 'trained_models_cv_v2_new'),
}

# Default training hyperparameters (matching the paper's main experiment)
DEFAULT_N_RUNS = 20
DEFAULT_N_EPOCHS = 10
PATCH_SIZE = 16
N_NEURONS = 144
CV_CONS = 'cv_constrained_v2'
CV_RATIO = 0.40
RECON_BUDGET = 0.60
LMDA = 5e-3


# ═══════════════════════════════════════════════════════════════════════════
# Test image generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_pink_noise_image(size=128, alpha=1.0, rng=None):
    """
    Generate a single 1/f^alpha (pink noise) image.

    Parameters
    ----------
    size : int
        Image side length (square).
    alpha : float
        Power-spectrum exponent.  alpha ≈ 1.0 matches natural-image
        amplitude spectra.
    rng : np.random.Generator, optional

    Returns
    -------
    img : ndarray (size, size), values roughly in [-1, 1], zero-mean.
    """
    if rng is None:
        rng = np.random.default_rng()
    # White noise in frequency domain
    f_real = rng.normal(0, 1, (size, size))
    f_imag = rng.normal(0, 1, (size, size))
    f_noise = f_real + 1j * f_imag

    # Frequency grid
    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs)
    f_mag = np.sqrt(fx**2 + fy**2)
    f_mag[0, 0] = 1.0 / size  # avoid division by zero at DC

    # Apply 1/f^alpha scaling
    f_scaled = f_noise * (1.0 / (f_mag ** alpha))

    # Inverse FFT → spatial domain
    img = np.real(np.fft.ifft2(f_scaled))

    # Normalise to zero mean, unit std (then clip outliers for stability)
    img = (img - img.mean()) / (img.std() + 1e-12)
    img = np.clip(img, -3.0, 3.0)
    return img


def generate_pink_noise_dataset(n_images=10, img_size=128, alpha=1.0,
                                patch_size=16, border=0, seed=42):
    """
    Generate a dataset of 1/f noise images and extract patches.

    Returns
    -------
    images : ndarray (img_size, img_size, n_images)
    patches : ndarray (n_patches, patch_size*patch_size)
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((img_size, img_size, n_images), dtype=np.float32)

    # Generate full images
    for i in range(n_images):
        images[:, :, i] = generate_pink_noise_image(img_size, alpha, rng)

    # Extract non-overlapping 16×16 patches (deterministic grid)
    ppr = img_size // patch_size  # patches per row
    patches_per_img = ppr * ppr
    all_patches = np.zeros((n_images * patches_per_img,
                             patch_size * patch_size), dtype=np.float32)

    for i_img in range(n_images):
        img = images[:, :, i_img]
        for row in range(ppr):
            for col in range(ppr):
                x0, y0 = row * patch_size, col * patch_size
                if x0 + patch_size <= img_size and y0 + patch_size <= img_size:
                    patch = img[x0:x0 + patch_size,
                                y0:y0 + patch_size]
                    idx = i_img * patches_per_img + row * ppr + col
                    all_patches[idx] = patch.ravel() - patch.mean()

    return images, all_patches


def generate_structured_patterns(patch_size=16, seed=42):
    """
    Generate the historical inputs unchanged, including the duplicate edge.

    Deduplicate scoring with exact_unique_inputs, not by changing these
    coordinates: centered edges would constitute a different experiment.

    Returns
    -------
    patterns : dict mapping name → ndarray (patch_size*patch_size,)
    """
    rng = np.random.default_rng(seed)
    patterns = {}
    xs = np.arange(patch_size)
    xv, yv = np.meshgrid(xs, xs)

    # ── Gratings at various orientations and frequencies ──
    for orient_deg in [0, 30, 45, 60, 90, 120, 135]:
        theta = np.deg2rad(orient_deg)
        for freq in [1, 2, 3]:
            # Spatial frequency in cycles per patch
            proj = np.cos(theta) * xv + np.sin(theta) * yv
            grating = np.cos(2 * np.pi * freq * proj / patch_size)
            grating = grating - grating.mean()
            patterns[f'grating_{orient_deg}deg_f{freq}'] = grating.ravel()

    # ── Edges (step functions at various orientations) ──
    for orient_deg in [0, 45, 90, 135]:
        theta = np.deg2rad(orient_deg)
        proj = np.cos(theta) * xv + np.sin(theta) * yv
        edge = np.where(proj > 0, 1.0, -1.0)
        edge = edge - edge.mean()
        patterns[f'edge_{orient_deg}deg'] = edge.ravel()

    # ── Bars (Gabor-like, sine-modulated Gaussian envelope) ──
    cx, cy = (patch_size - 1) / 2, (patch_size - 1) / 2
    for orient_deg in [0, 45, 90, 135]:
        theta = np.deg2rad(orient_deg)
        xr = (xv - cx) * np.cos(theta) + (yv - cy) * np.sin(theta)
        yr = -(xv - cx) * np.sin(theta) + (yv - cy) * np.cos(theta)
        sigma_x, sigma_y = 3.0, 6.0
        gauss = np.exp(-0.5 * (xr**2 / sigma_x**2 + yr**2 / sigma_y**2))
        bar = gauss * np.cos(2 * np.pi * xr / 4.0)
        bar = bar - bar.mean()
        patterns[f'bar_{orient_deg}deg'] = bar.ravel()

    # ── Centre-surround (difference of Gaussians) ──
    r2 = (xv - cx)**2 + (yv - cy)**2
    dog = np.exp(-r2 / (2 * 2.0**2)) - 0.7 * np.exp(-r2 / (2 * 5.0**2))
    dog = dog - dog.mean()
    patterns['centre_surround'] = dog.ravel()

    # ── Random dot pattern ──
    dots = rng.uniform(-1, 1, (patch_size, patch_size))
    dots = dots - dots.mean()
    patterns['random_dots'] = dots.ravel()

    # Normalise all patterns to unit L2 norm (consistent with whitened patches)
    for name in patterns:
        nrm = np.linalg.norm(patterns[name])
        if nrm > 1e-12:
            patterns[name] = patterns[name] / nrm

    return patterns


# ═══════════════════════════════════════════════════════════════════════════
# Model training (if needed)
# ═══════════════════════════════════════════════════════════════════════════

def method_models_exist(method, n_runs, n_epochs):
    """Check whether a method has all requested final-epoch checkpoints."""
    model_dir = MODEL_DIRS[method]
    if not os.path.isdir(model_dir):
        return False
    for k in range(n_runs):
        pth = os.path.join(model_dir, f'ckpt-{n_epochs}-{k}-1.pth')
        if not os.path.exists(pth):
            return False
    return True


def models_exist(n_runs, n_epochs):
    """Check whether trained model checkpoints exist for all three methods."""
    return all(method_models_exist(m, n_runs, n_epochs) for m in MODEL_DIRS)


def train_models(n_runs=DEFAULT_N_RUNS, n_epochs=DEFAULT_N_EPOCHS):
    """Train all three models via train_new.py subprocess calls."""
    print("\n" + "=" * 70)
    print("  TRAINING: Models not found — launching train_new.py")
    print("=" * 70)

    configs = [
        ('null',   'non'),
        ('cauchy', 's'),
        ('cv_m',   CV_CONS),
    ]

    for method, cons in configs:
        if method_models_exist(method, n_runs, n_epochs):
            print(f"\n── {method} checkpoints already present; skipping ──")
            continue
        print(f"\n── Training {method} ({cons}) ──")
        cmd = [
            sys.executable, os.path.join(BASE_DIR, 'train_new.py'),
            '--cons', cons,
            '--n_runs', str(n_runs),
            '--run_start', '0',
            '--epoch', str(n_epochs),
            '--reg', str(LMDA),
            '--cv_ratio', str(CV_RATIO),
            '--recon-budget', str(RECON_BUDGET),
        ]
        subprocess.run(cmd, check=True, cwd=BASE_DIR)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# ISTA inference with a given dictionary
# ═══════════════════════════════════════════════════════════════════════════

def ista_infer(U_weight, patches, lmda=5e-3, r_lr=1e-2,
               max_iter=100, device=None):
    """
    Run ISTA inference to find sparse coefficients for input patches
    using a fixed dictionary U.

    This is a plain LASSO/ISTA solver — gradient step on the reconstruction
    term followed by the soft-thresholding proximal operator (the L1 prox).
    No Cauchy or CV penalty is applied: inference is deliberately identical
    across all three dictionaries so the comparison isolates *dictionary
    quality* from inference-time regularization.

    Parameters
    ----------
    U_weight : torch.Tensor (M², K) — dictionary weight matrix
               (PyTorch nn.Linear convention: weight = [out_features, in_features])
    patches : torch.Tensor (N, M²) — input patches
    lmda : float — soft-thresholding parameter
    r_lr : float — ISTA step size
    max_iter : int — maximum ISTA iterations
    device : torch.device

    Returns
    -------
    R : torch.Tensor (N, K) — sparse coefficients
    converged : bool — whether the batch-level relative change fell below tol
    n_iters : int — iterations actually run
    """
    if device is None:
        device = torch.device('cpu')

    U_weight = U_weight.to(device)
    patches = patches.to(device)
    N, M2 = patches.shape
    M2_u, K = U_weight.shape
    assert M2 == M2_u, f"Patch dim {M2} != weight dim {M2_u}"

    R = torch.zeros((N, K), requires_grad=True, device=device)
    optim = torch.optim.SGD([R], lr=r_lr)

    converged = False
    it = 0

    while not converged and it < max_iter:
        it += 1
        old_R = R.clone().detach()
        optim.zero_grad()

        # U has shape [M², K]; pred = R @ U.T = [N, K] @ [K, M²] = [N, M²]
        pred = R @ U_weight.T
        recon = ((patches - pred) ** 2).sum()
        if not torch.isfinite(recon):
            raise FloatingPointError(
                f"ISTA reconstruction loss became non-finite at iter {it}"
            )

        # Gradient step on the smooth (reconstruction) term only.
        recon.backward()
        optim.step()

        # Proximal operator: soft-thresholding (the L1 prox).
        with torch.no_grad():
            R.data = (F.relu(R.data - lmda)
                      - F.relu(-R.data - lmda))
            if not torch.isfinite(R).all():
                raise FloatingPointError(
                    f"ISTA coefficients became non-finite at iter {it}"
                )

        with torch.no_grad():
            change = torch.norm(R - old_R) / (torch.norm(old_R) + 1e-8)
            converged = (change < 0.01).item()

    return R.detach(), converged, it


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_dictionary(U_weight, patches, patch_size=16, device=None,
                        return_per_patch=False, scoring_indices=None):
    """
    Evaluate a dictionary on a set of patches.

    Parameters
    ----------
    U_weight : ndarray (M², K)
    patches : ndarray (N, M²)

    Returns
    -------
    results : dict with keys: RE, SSIM, ICE, mean_a, cv_pop, independence,
                               converged_rate, avg_iters
    """
    if device is None:
        device = torch.device('cpu')

    M2, K = U_weight.shape  # U is [M², K] (PyTorch Linear convention)
    N = patches.shape[0]
    if not np.isfinite(U_weight).all():
        raise FloatingPointError("Non-finite dictionary weights")
    if not np.isfinite(patches).all():
        raise FloatingPointError("Non-finite evaluation patches")

    U_t = torch.tensor(U_weight, dtype=torch.float32)
    P_t = torch.tensor(patches, dtype=torch.float32)

    # Run ISTA inference
    R, converged, n_iters = ista_infer(U_t, P_t, device=device)

    R_np = R.cpu().numpy()
    U_np = U_weight
    if not np.isfinite(R_np).all():
        raise FloatingPointError("Non-finite coefficients")

    # Reconstruction: U is [M², K], R is [N, K], so Î = R @ U.T = [N, M²]
    recon_np = R_np @ U_np.T  # [N, K] @ [K, M²] = [N, M²]
    if not np.isfinite(recon_np).all():
        raise FloatingPointError("Non-finite reconstruction")

    # Keep the original inference batch: dropping aliases before ISTA would
    # change its batch-relative stopping criterion for the retained inputs.
    n_inference = N
    if scoring_indices is not None:
        indices = np.asarray(scoring_indices, dtype=int)
        if (indices.ndim != 1 or len(indices) == 0 or len(set(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= N)):
            raise ValueError('Scoring indices must be unique valid input indices')
        patches, recon_np, R_np = patches[indices], recon_np[indices], R_np[indices]
        N = len(indices)

    # ── Per-patch metrics ──
    re_vals = np.zeros(N)
    ssim_vals = np.zeros(N)
    mean_a_vals = np.zeros(N)

    for i in range(N):
        orig = patches[i].reshape(patch_size, patch_size)
        rec = recon_np[i].reshape(patch_size, patch_size)

        # RE
        re_vals[i] = (np.linalg.norm(orig - rec)
                      / (np.linalg.norm(orig) + 1e-12))

        # SSIM
        ssim_vals[i] = reference_ssim(orig, rec)

        # Mean |a|
        mean_a_vals[i] = np.mean(np.abs(R_np[i]))

    # ── Aggregate metrics ──
    re_mean = np.mean(re_vals)
    re_std = np.std(re_vals)
    ssim_mean = np.mean(ssim_vals)
    mean_a = np.mean(mean_a_vals)

    # RAE = -log2(RE) / mean_a. The result key stays "ICE" for archive compatibility.
    ice = -np.log2(re_mean + 1e-12) / (mean_a + 1e-12)

    # Population CV
    R_abs = np.abs(R_np)
    cv_pop = np.mean([np.std(R_abs[i]) / (np.mean(R_abs[i]) + 1e-12)
                       for i in range(N)])

    # Basis independence
    dec = decorrelation(U_np)  # [K, K]
    independence = np.sum(dec) / (K * (K - 1)) * 100

    results = {
        'RE': re_mean,
        'RE_std': re_std,
        'SSIM': ssim_mean,
        'ICE': ice,
        'mean_a': mean_a,
        'cv_pop': cv_pop,
        'independence': independence,
        'converged_rate': float(converged),
        'avg_iters': n_iters,
        'n_scored_inputs': N,
        'n_inference_inputs': n_inference,
        'n_constant_references': int(np.sum(np.ptp(patches, axis=1) == 0)),
    }
    bad_metrics = [
        key for key, value in results.items()
        if not np.isfinite(value)
    ]
    if bad_metrics:
        raise FloatingPointError(
            "Non-finite evaluation metrics: " + ", ".join(bad_metrics)
        )
    if return_per_patch:
        results['per_patch'] = [
            {
                'patch_index': i,
                'RE': float(re_vals[i]),
                'SSIM': float(ssim_vals[i]),
                'mean_a': float(mean_a_vals[i]),
                'cv_pop': float(np.std(R_abs[i]) / (np.mean(R_abs[i]) + 1e-12)),
            }
            for i in range(N)
        ]
    return results


def count_checkpoints(model_dir, n_epochs, run_idx):
    """Count available checkpoints for a given run."""
    count = 0
    # Checkpoints are numbered 1..N (one per batch in the last epoch)
    for j in range(1, 100):  # safe upper bound
        if os.path.exists(os.path.join(model_dir,
                                        f'ckpt-{n_epochs}-{run_idx}-{j}.pth')):
            count += 1
        elif count > 0:
            break  # sequential: stop at first gap
    return count


def evaluate_across_runs(method, model_dir, patches, n_runs, n_epochs,
                         device=None, scoring_indices=None):
    """
    Evaluate a method across all training runs.

    For each run, loads all available checkpoints (one per batch from the
    last training epoch), evaluates each dictionary on the full test set,
    and averages the results within the run.

    Returns list of per-run result dicts.
    """
    if device is None:
        device = torch.device('cpu')

    results = []
    patch_size = int(np.sqrt(patches.shape[1]))

    for k in range(n_runs):
        n_ckpt = count_checkpoints(model_dir, n_epochs, k)
        if n_ckpt == 0:
            print(f"  [{method}] run={k}: no checkpoints found, skipping")
            continue

        run_results = []
        n_skipped = 0
        for j in range(1, n_ckpt + 1):
            pth = os.path.join(model_dir, f'ckpt-{n_epochs}-{k}-{j}.pth')
            if not os.path.exists(pth):
                continue

            net = torch.load(pth, map_location='cpu', weights_only=False)
            U = net.U.weight.detach().cpu().numpy()

            try:
                res = evaluate_dictionary(U, patches, patch_size, device,
                                          scoring_indices=scoring_indices)
            except FloatingPointError as exc:
                n_skipped += 1
                print(f"  [{method}] run={k} ckpt={j}: skipped ({exc})")
                continue
            run_results.append(res)

        if not run_results:
            print(f"  [{method}] run={k}: no finite checkpoint evaluations, skipping")
            continue

        # Average across checkpoints within this run
        run_avg = {}
        for key in run_results[0]:
            vals = [r[key] for r in run_results]
            run_avg[key] = float(np.mean(vals))
        run_avg['run'] = k
        run_avg['n_checkpoints'] = len(run_results)
        run_avg['n_skipped_checkpoints'] = n_skipped
        results.append(run_avg)

        print(f"  [{method}] run={k}: RE={run_avg['RE']:.4f}, "
              f"SSIM={run_avg['SSIM']:.4f}, ICE={run_avg['ICE']:.2f}, "
              f"indep={run_avg['independence']:.1f}% "
              f"({len(run_results)} ckpts)")

    return results


def evaluate_pattern_level(model_dirs, pattern_names, patches, n_runs, n_epochs,
                           device=None, scoring_indices=None):
    """
    Paired pattern-level summary for structured stimuli.

    Returns one row per (method, pattern), after averaging over valid training
    runs and last-epoch checkpoints.  This makes the hand-crafted pattern, not
    the random seed, the unit for the directional CVM-vs-Null statement.
    """
    if device is None:
        device = torch.device('cpu')
    patch_size = int(np.sqrt(patches.shape[1]))
    per_run_pattern = []

    for method, model_dir in model_dirs.items():
        for k in range(n_runs):
            n_ckpt = count_checkpoints(model_dir, n_epochs, k)
            if n_ckpt == 0:
                continue
            ckpt_rows = []
            for j in range(1, n_ckpt + 1):
                pth = os.path.join(model_dir, f'ckpt-{n_epochs}-{k}-{j}.pth')
                if not os.path.exists(pth):
                    continue
                net = torch.load(pth, map_location='cpu', weights_only=False)
                U = net.U.weight.detach().cpu().numpy()
                try:
                    res = evaluate_dictionary(
                        U, patches, patch_size, device,
                        return_per_patch=True,
                        scoring_indices=scoring_indices,
                    )
                except FloatingPointError as exc:
                    print(f"  [{method}] run={k} ckpt={j}: "
                          f"skipped pattern eval ({exc})")
                    continue
                ckpt_rows.extend(res['per_patch'])
            if not ckpt_rows:
                continue
            df = pd.DataFrame(ckpt_rows)
            for idx, name in enumerate(pattern_names):
                sub = df[df['patch_index'] == idx]
                per_run_pattern.append({
                    'type': method,
                    'run': k,
                    'pattern': name,
                    'RE': float(sub['RE'].mean()),
                    'SSIM': float(sub['SSIM'].mean()),
                    'mean_a': float(sub['mean_a'].mean()),
                })

    if not per_run_pattern:
        return [], {}

    df_run = pd.DataFrame(per_run_pattern)
    pattern_rows = (df_run.groupby(['type', 'pattern'])[['RE', 'SSIM', 'mean_a']]
                    .mean().reset_index().to_dict('records'))
    piv = pd.DataFrame(pattern_rows).pivot(index='pattern', columns='type',
                                           values='SSIM')
    if {'null', 'cv_m'}.issubset(piv.columns):
        diff = piv['cv_m'] - piv['null']
        paired_summary = {
            'metric': 'SSIM',
            'unit': 'structured pattern',
            'n_patterns': int(diff.notna().sum()),
            'cvm_gt_null': int((diff > 0).sum()),
            'cvm_ge_null': int((diff >= 0).sum()),
            'mean_diff': float(diff.mean()),
            'median_diff': float(diff.median()),
            'min_diff': float(diff.min()),
            'max_diff': float(diff.max()),
        }
    else:
        paired_summary = {}
    return pattern_rows, paired_summary


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def add_stat_annotation(ax, data, x, y, pairs, order, paired_unit=None):
    """Annotate raw mean differences; inferential results are in the JSON tables."""
    y_max = data[y].max()
    y_range = data[y].max() - data[y].min()
    if y_range == 0:
        y_range = y_max + 0.01
    h = 0.05 * y_range
    offset = 0.08 * y_range
    pos_map = {name: i for i, name in enumerate(order)}

    for ki, (g1, g2) in enumerate(pairs):
        v1 = data.loc[data[x] == g1, y].values
        v2 = data.loc[data[x] == g2, y].values
        if paired_unit is not None:
            v1, v2 = paired_values(data.to_dict('records'), y, g1, g2, paired_unit)
        if len(v1) < 2 or len(v2) < 2:
            continue
        difference = float(np.mean(v2) - np.mean(v1))
        x1, x2 = pos_map[g1], pos_map[g2]
        y_bar = y_max + offset + ki * (h + offset)
        ax.plot([x1, x1, x2, x2],
                [y_bar, y_bar + h, y_bar + h, y_bar],
                lw=1.0, color='k')
        ax.text((x1 + x2) / 2, y_bar + h, f"delta={difference:+.3g}",
                ha='center', va='bottom', fontsize=9)


def plot_metrics_boxplot(results_df, test_name, output_dir):
    """Generate boxplot panels for RE, SSIM, ICE with stats."""
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams["axes.labelsize"] = 14

    palette = {'null': '#0073C2FF', 'cauchy': '#0073C2FF', 'cv_m': '#EFC000FF'}
    type_order = ['null', 'cauchy', 'cv_m']
    label_map = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
    box_pairs = [("null", "cauchy"), ("null", "cv_m"), ("cauchy", "cv_m")]

    features = ['RE', 'SSIM', 'ICE']
    ylabels = {
        'RE': 'Relative Reconstruction Error (RE)',
        'SSIM': 'Structural Similarity (SSIM)',
        'ICE': r'RAE score $= -\log_2(\mathrm{RE})/\bar{a}$',
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120, facecolor='w')

    for ax, feature in zip(axes, features):
        # Map type to readable labels for x-axis
        plot_df = results_df.copy()
        plot_df['label'] = plot_df['type'].map(label_map)
        label_order = [label_map[t] for t in type_order]
        label_palette = {label_map[k]: v for k, v in palette.items()}

        sns.boxplot(x='label', y=feature, data=plot_df, order=label_order,
                    hue='label', palette=label_palette, legend=False, ax=ax)

        # Stats: use the original 'type' column for grouping
        stats_df = results_df.copy()
        # Map pair labels for the stat annotation (it uses original type keys)
        add_stat_annotation(ax, stats_df, 'type', feature,
                           box_pairs, type_order)

        ax.set_ylabel(ylabels.get(feature, feature))
        ax.set_xlabel('')

    fig.suptitle(f'Generalization Test — {test_name}',
                 fontsize=16, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_dir, 'metrics_boxplot.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_reconstruction_comparison(method_dictionaries, test_patches,
                                   test_name, output_dir, n_examples=8):
    """Show original vs reconstructed patches side-by-side for each method."""
    patch_size = int(np.sqrt(test_patches.shape[1]))
    n_show = min(n_examples, test_patches.shape[0])

    label_map = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
    n_methods = len(method_dictionaries)

    fig, axes = plt.subplots(n_show, 1 + n_methods,
                             figsize=(2 * (1 + n_methods), 2 * n_show),
                             dpi=120, facecolor='w')

    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        patch = test_patches[i].reshape(patch_size, patch_size)
        axes[i, 0].imshow(patch, cmap='gray', vmin=-2, vmax=2)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=10)

        for j, (method, U) in enumerate(method_dictionaries.items()):
            U_t = torch.tensor(U, dtype=torch.float32)
            P_t = torch.tensor(test_patches[i:i+1], dtype=torch.float32)
            R, _, _ = ista_infer(U_t, P_t)
            recon = (R.cpu().numpy() @ U.T).reshape(patch_size, patch_size)

            vmin = min(patch.min(), recon.min())
            vmax = max(patch.max(), recon.max())
            axes[i, j + 1].imshow(recon, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(label_map.get(method, method),
                                         fontsize=10)

    fig.suptitle(f'Reconstruction Comparison — {test_name}',
                 fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_dir, 'reconstruction_comparison.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_dictionary_grid(method_dictionaries, output_dir):
    """Plot the dictionary basis functions for each method as an image grid."""
    label_map = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
    patch_size = 16
    grid_size = 12  # show 12×12 = 144 basis functions

    for method, U in method_dictionaries.items():
        # U has shape [M², K]; each column is a basis function
        M2, K = U.shape
        n_show = min(K, grid_size * grid_size)
        fig, axes = plt.subplots(grid_size, grid_size,
                                 figsize=(12, 12), dpi=100, facecolor='w')

        for i in range(grid_size * grid_size):
            row, col = i // grid_size, i % grid_size
            ax = axes[row, col]
            if i < n_show:
                atom = U[:, i].reshape(patch_size, patch_size)
                atom = (atom - atom.min()) / (atom.max() - atom.min() + 1e-8)
                ax.imshow(atom, cmap='gray')
            ax.axis('off')

        fig.suptitle(f'Basis Functions — {label_map.get(method, method)}',
                     fontsize=16, y=1.01)
        fig.tight_layout()

        out_path = os.path.join(output_dir,
                                f'dictionary_{method}.pdf')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out_path}")


def plot_sample_images(images, test_name, output_dir, n_show=4):
    """Plot sample test images and extracted patches."""
    img_size = images.shape[0]
    n_images = images.shape[2]
    n_show = min(n_show, n_images)

    fig, axes = plt.subplots(n_show, 1, figsize=(6, 2 * n_show),
                             dpi=120, facecolor='w')
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        im = axes[i].imshow(images[:, :, i], cmap='gray')
        axes[i].set_title(f'Test image {i+1} (1/f noise, {img_size}×{img_size})',
                          fontsize=10)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    fig.suptitle(f'Sample Test Images — {test_name}', fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_dir, 'sample_images.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def json_safe_rows(rows):
    return [
        {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in r.items()
        }
        for r in rows
    ]


def print_summary_table(all_results, test_name):
    """Print group means with bootstrap 95% CIs and pairwise effect sizes."""
    label_map = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
    type_order = ['null', 'cauchy', 'cv_m']
    report_cols = ['RE', 'SSIM', 'ICE', 'mean_a', 'independence']
    box_pairs = [("null", "cauchy"), ("null", "cv_m"), ("cauchy", "cv_m")]
    print_effect_table(all_results, report_cols, type_order, label_map,
                       box_pairs, test_name, unit_label='training run')


def run_test(test_name, get_patches_fn, device, n_runs, n_epochs):
    """
    Run a generalization test.

    Parameters
    ----------
    test_name : str — 'Pink Noise' or 'Structured Patterns'
    get_patches_fn : callable — returns (images_or_None, patches_ndarray)
                             or (images_or_None, patches_ndarray, names)
    device : torch.device
    n_runs, n_epochs : int
    """
    output_dir = os.path.join(OUTPUT_DIR, test_name.lower().replace(' ', '_'))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'─' * 70}")
    print(f"  TEST: {test_name}")
    print(f"{'─' * 70}")

    # Generate test data
    print("  Generating test data...")
    test_data = get_patches_fn()
    if len(test_data) == 3:
        images, patches, pattern_names = test_data
    else:
        images, patches = test_data
        pattern_names = None
    print(f"  Test patches shape: {patches.shape}")
    scoring_indices = None
    if pattern_names is not None:
        scoring_indices, manifest = exact_unique_inputs(pattern_names, patches)
        pattern_names = manifest['retained_names']
        if test_name == 'Structured Patterns' and (
                len(scoring_indices) != 30 or manifest['aliases'] != {'edge_90deg': 'edge_45deg'}):
            raise ValueError('Historical structured bank changed; refusing silent experiment replacement')
        manifest_path = os.path.join(output_dir, 'input_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                validate_frozen_manifest(manifest, json.load(f))
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    # Visualise sample images (if images available)
    if images is not None:
        plot_sample_images(images, test_name, output_dir)

    # Collect dictionaries from best run of each method
    method_dictionaries = {}
    all_results = []

    for method, model_dir in MODEL_DIRS.items():
        print(f"\n  Evaluating {method}...")
        results = evaluate_across_runs(method, model_dir, patches,
                                       n_runs, n_epochs, device=device,
                                       scoring_indices=scoring_indices)
        for r in results:
            r['type'] = method
        all_results.extend(results)

        # Load one representative dictionary for visualisation
        # (use the first available checkpoint from run 0)
        pth = os.path.join(model_dir, f'ckpt-{n_epochs}-0-1.pth')
        if os.path.exists(pth):
            net = torch.load(pth, map_location='cpu', weights_only=False)
            method_dictionaries[method] = net.U.weight.detach().cpu().numpy()
        else:
            print(f"  [WARN] No checkpoint found for {method} dictionary viz")

    # ── Build DataFrame ──
    df = pd.DataFrame(all_results)
    if df.empty:
        print("  [ERROR] No results collected — skipping plots.")
        return all_results

    # ── Plots ──
    print(f"\n  Generating plots for {test_name}...")
    plot_metrics_boxplot(df, test_name, output_dir)

    if len(method_dictionaries) >= 2 and patches.shape[0] > 0:
        plot_reconstruction_comparison(method_dictionaries, patches,
                                       test_name, output_dir)
        plot_dictionary_grid(method_dictionaries, output_dir)

    # ── Summary table ──
    print_summary_table(all_results, test_name)

    # ── Save numerical results ──
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(json_safe_rows(all_results), f, indent=2)
    print(f"  Saved → {summary_path}")

    if pattern_names is not None:
        pattern_rows, paired_summary = evaluate_pattern_level(
            MODEL_DIRS, pattern_names, patches, n_runs, n_epochs,
            device=device,
            scoring_indices=scoring_indices,
        )
        pattern_path = os.path.join(output_dir, 'summary_by_pattern.json')
        pair_path = os.path.join(output_dir, 'pattern_pair_summary.json')
        with open(pattern_path, 'w') as f:
            json.dump(json_safe_rows(pattern_rows), f, indent=2)
        with open(pair_path, 'w') as f:
            json.dump(paired_summary, f, indent=2)
        if paired_summary:
            print("  Pattern-level paired SSIM: "
                  f"CVM > Null in {paired_summary['cvm_gt_null']}/"
                  f"{paired_summary['n_patterns']} patterns")
        print(f"  Saved → {pattern_path} + {pair_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Generalization test for Fig6 sparse coding models')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training; use existing checkpoints')
    parser.add_argument('--n-runs', type=int, default=DEFAULT_N_RUNS,
                        help=f'Number of training runs (default: {DEFAULT_N_RUNS})')
    parser.add_argument('--n-epochs', type=int, default=DEFAULT_N_EPOCHS,
                        help=f'Training epochs (default: {DEFAULT_N_EPOCHS})')
    parser.add_argument('--n-test-images', type=int, default=10,
                        help='Number of 1/f noise test images (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for test image generation (default: 42)')
    parser.add_argument('--pink-noise-only', action='store_true',
                        help='Only run the pink noise test')
    parser.add_argument('--patterns-only', action='store_true',
                        help='Only run the structured patterns test')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")

    # ── Train models if needed ──
    if not args.skip_train:
        if not models_exist(args.n_runs, args.n_epochs):
            print("\nModels not found — starting training...")
            train_models(args.n_runs, args.n_epochs)
        else:
            print("\nModels already exist — skipping training.")
    else:
        if not models_exist(args.n_runs, args.n_epochs):
            raise FileNotFoundError('--skip-train requires all saved checkpoints')
        else:
            print("\nSkipping training (--skip-train).")

    # ── Run tests ──
    all_test_results = {}

    if not args.patterns_only:
        # ── Test 1: Pink noise images ──
        def get_pink_noise():
            images, patches = generate_pink_noise_dataset(
                n_images=args.n_test_images,
                img_size=128,
                alpha=1.0,
                patch_size=PATCH_SIZE,
                seed=args.seed,
            )
            return images, patches

        all_test_results['pink_noise'] = run_test(
            'Pink Noise', get_pink_noise, device,
            args.n_runs, args.n_epochs,
        )

    if not args.pink_noise_only:
        # ── Test 2: Structured patterns ──
        def get_structured():
            patterns = generate_structured_patterns(
                patch_size=PATCH_SIZE, seed=args.seed)
            # Stack into array
            names = sorted(patterns.keys())
            patches = np.stack([patterns[n] for n in names], axis=0)
            return None, patches, names  # no full images to show

        all_test_results['structured_patterns'] = run_test(
            'Structured Patterns', get_structured, device,
            args.n_runs, args.n_epochs,
        )

    # ── Combined report ──
    print(f"\n{'=' * 70}")
    print("  ALL TESTS COMPLETE")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    # Print a concise final summary across all tests
    for test_name, results in all_test_results.items():
        if not results:
            continue
        best_method = max(
            ['null', 'cauchy', 'cv_m'],
            key=lambda m: np.mean([r['ICE'] for r in results
                                   if r['type'] == m])
        )
        best_ssim = max(
            ['null', 'cauchy', 'cv_m'],
            key=lambda m: np.mean([r['SSIM'] for r in results
                                   if r['type'] == m])
        )
        best_re = min(
            ['null', 'cauchy', 'cv_m'],
            key=lambda m: np.mean([r['RE'] for r in results
                                   if r['type'] == m])
        )
        label_map = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
        print(f"\n  {test_name}:")
        print(f"    Best RE:   {label_map[best_re]}")
        print(f"    Best SSIM: {label_map[best_ssim]}")
        print(f"    Best ICE:  {label_map[best_method]}")


if __name__ == '__main__':
    main()
