"""
Pareto scan: train CVM with different cv_ratio values, then plot
the RE–CV and RE–ICE frontier to find a better operating point.

Usage:
    cd Fig6
    python pareto_scan.py            # train + evaluate + plot
    python pareto_scan.py --eval-only # skip training, re-evaluate existing runs
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # suppress duplicate OpenMP warning on Windows
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
import torch
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR  = os.path.join(BASE_DIR, 'record')
DATA_PATH   = os.path.join(BASE_DIR, 'src', 'model', 'IMAGES.mat')

CV_RATIOS   = [0.02, 0.04, 0.06, 0.08, 0.10]   # 0.10 = current default
N_RUNS      = 2    # runs per setting (quick scan; use 5+ for paper)
N_EPOCHS    = 10   # must match USE_NEW checkpoint convention in result_new.py


def tag(cv_ratio: float) -> str:
    return f"r{cv_ratio:.2f}"


def suffix(cv_ratio: float) -> str:
    return f"_cv_16_nl_new_{tag(cv_ratio)}"


# ── Training ───────────────────────────────────────────────────────────────────
def run_training(cv_ratios, n_runs, n_epochs):
    for r in cv_ratios:
        print(f"\n{'='*60}")
        print(f"Training CVM  cv_ratio={r}  (tag={tag(r)})")
        print(f"{'='*60}")
        cmd = [
            sys.executable, os.path.join(BASE_DIR, 'train_new.py'),
            '--cons', 'cv',
            '--cv_ratio', str(r),
            '--n_runs',   str(n_runs),
            '--run_start','0',
            '--epoch',    str(n_epochs),
            '--tag',      tag(r),
        ]
        subprocess.run(cmd, check=True, cwd=BASE_DIR)


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_one(cv_ratio, n_runs, n_epochs, X, img_size, n_img):
    """Return a dict with mean metrics for a given cv_ratio."""
    sfx  = suffix(cv_ratio)
    model_dir = os.path.join(RECORD_DIR, 'trained_models' + sfx)

    re_list, cv_list, cost_list = [], [], []

    for k in range(n_runs):
        record_path = os.path.join(RECORD_DIR, f'record{sfx}_{k}.mat')
        if not os.path.exists(record_path):
            print(f"  [SKIP] record not found: {record_path}")
            continue
        record = scio.loadmat(record_path)

        re_acc = 0.0; cost_acc = 0.0; cv_acc = 0.0; n_loaded = 0

        for j in range(n_img):
            img = X[:, :, j]
            pth = os.path.join(model_dir, f'ckpt-{n_epochs}-{k}-{j+1}.pth')
            if not os.path.exists(pth):
                print(f"  [SKIP] model not found: {pth}")
                continue

            net = torch.load(pth, map_location='cpu')
            w   = torch.abs(net.R).detach().numpy()   # [n_patch, K]

            # cost = mean |activation|
            cost_acc += np.mean(w)

            # population CV (per-patch, averaged)
            cv_pp = 0.0
            for i in range(w.shape[0]):
                mu_i = np.mean(w[i])
                if mu_i > 1e-12:
                    cv_pp += np.std(w[i]) / mu_i
            cv_acc += cv_pp / w.shape[0]

            # reconstruct full image
            recon = net.U(net.R).detach().numpy()
            patch_w = int(np.sqrt(recon.shape[1]))
            img_rec = np.zeros([img_size, img_size])
            ppr = img_size // patch_w
            for i in range(recon.shape[0]):
                patch = recon[i].reshape(patch_w, patch_w)
                x0 = (i % ppr) * patch_w;  y0 = (i // ppr) * patch_w
                if x0 + patch_w <= img_size and y0 + patch_w <= img_size:
                    img_rec[x0:x0+patch_w, y0:y0+patch_w] = patch

            re_acc += np.linalg.norm(img - img_rec) / (np.linalg.norm(img) + 1e-12)
            n_loaded += 1

        if n_loaded == 0:
            continue

        re_list.append(re_acc / n_loaded)
        cv_list.append(cv_acc / n_loaded)
        cost_list.append(cost_acc / n_loaded)

    if not re_list:
        return None

    re_m   = np.mean(re_list);   re_s   = np.std(re_list)
    cv_m   = np.mean(cv_list);   cv_s   = np.std(cv_list)
    cost_m = np.mean(cost_list)

    ice = -np.log2(re_m + 1e-12) / (cost_m + 1e-12)
    print(f"  cv_ratio={cv_ratio:.2f}  RE={re_m:.4f}±{re_s:.4f}  "
          f"CV={cv_m:.3f}±{cv_s:.3f}  ā={cost_m:.4f}  ICE={ice:.2f}")
    return dict(cv_ratio=cv_ratio, RE=re_m, RE_std=re_s,
                CV=cv_m, CV_std=cv_s, cost=cost_m, ICE=ice)


def run_evaluation(cv_ratios, n_runs, n_epochs):
    X      = loadmat(DATA_PATH)['IMAGES']
    img_sz = X.shape[0]
    n_img  = X.shape[2]

    rows = []
    for r in cv_ratios:
        res = evaluate_one(r, n_runs, n_epochs, X, img_sz, n_img)
        if res is not None:
            rows.append(res)

    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_pareto(df, baselines=None):
    """
    baselines: dict  {'null': (RE, CV, ICE, cost), 'cauchy': (...)}
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=120)

    cmap   = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, len(df)))

    # ── Panel 1: RE vs CV ──
    ax = axes[0]
    for i, row in df.iterrows():
        ax.errorbar(row['CV'], row['RE'],
                    xerr=row['CV_std'], yerr=row['RE_std'],
                    fmt='o', color=colors[i], markersize=8,
                    label=f"cv_ratio={row['cv_ratio']:.2f}")
    if baselines:
        for name, (re, cv, ice, cost) in baselines.items():
            ax.axhline(re, ls='--', lw=1, alpha=0.6, label=f'{name} RE={re:.3f}')
            ax.axvline(cv, ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('Population CV', fontsize=13)
    ax.set_ylabel('Relative reconstruction error (RE)', fontsize=13)
    ax.set_title('Pareto frontier: RE vs CV', fontsize=13)
    ax.legend(fontsize=9, frameon=False)

    # Panel 2: RAE vs cv_ratio (internal arrays retain the legacy ICE key).
    ax = axes[1]
    ax.errorbar(df['cv_ratio'], df['ICE'], fmt='s-', color='steelblue',
                markersize=8, linewidth=1.5)
    if baselines:
        for name, (re, cv, ice, cost) in baselines.items():
            ax.axhline(ice, ls='--', lw=1, alpha=0.6, label=f'{name} ICE={ice:.2f}')
    ax.set_xlabel('cv_ratio', fontsize=13)
    ax.set_ylabel(r'RAE score $= -\log_2(\mathrm{RE})/\bar{a}$', fontsize=11)
    ax.set_title('RAE vs cv_ratio', fontsize=13)
    if baselines:
        ax.legend(fontsize=9, frameon=False)

    # ── Panel 3: RE and ā vs cv_ratio ──
    ax = axes[2]
    ax2 = ax.twinx()
    l1, = ax.plot(df['cv_ratio'], df['RE'],  'o-', color='tomato',   label='RE')
    l2, = ax2.plot(df['cv_ratio'], df['cost'], 's--', color='steelblue', label=r'$\bar{a}$ (cost)')
    if baselines:
        for name, (re, cv, ice, cost) in baselines.items():
            ax.axhline(re,   ls=':', color='tomato',    alpha=0.5)
            ax2.axhline(cost, ls=':', color='steelblue', alpha=0.5)
    ax.set_xlabel('cv_ratio', fontsize=13)
    ax.set_ylabel('RE', fontsize=13, color='tomato')
    ax2.set_ylabel(r'$\bar{a}$', fontsize=13, color='steelblue')
    ax.set_title(r'RE and $\bar{a}$ vs cv_ratio', fontsize=13)
    ax.legend(handles=[l1, l2], fontsize=9, frameon=False)

    fig.tight_layout()
    out = os.path.join(BASE_DIR, 'pareto_scan.pdf')
    fig.savefig(out)
    print(f"\nSaved → {out}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--eval-only', action='store_true',
                   help="Skip training, only evaluate existing checkpoints")
    p.add_argument('--n-runs',   type=int, default=N_RUNS)
    p.add_argument('--n-epochs', type=int, default=N_EPOCHS)
    args = p.parse_args()

    if not args.eval_only:
        run_training(CV_RATIOS, args.n_runs, args.n_epochs)

    print("\nEvaluating all cv_ratio settings...")
    df = run_evaluation(CV_RATIOS, args.n_runs, args.n_epochs)

    if df.empty:
        print("No results found. Run training first (drop --eval-only).")
        sys.exit(1)

    print("\n── Summary ─────────────────────────────────────────────")
    print(df[['cv_ratio', 'RE', 'CV', 'cost', 'ICE']].to_string(index=False,
                                                                  float_format='{:.4f}'.format))

    # Baseline values from current result_new.py run
    baselines = {
        'null':   (0.6110, 2.6288, 12.3003, 0.0578),
        'cauchy': (0.7020, 2.4798, 12.1754, 0.0419),
    }
    plot_pareto(df, baselines)
