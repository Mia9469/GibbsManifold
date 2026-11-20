"""
Compare Option 1 / 2 / 3 for CVM training:

  Option 1  – baseline cv_ratio scan (results already in pareto_scan.py)
  Option 2  – lower lmda (soft-threshold) + various cv_ratio
              lets more small activations participate → CV from distribution shape
  Option 3  – constrained: apply CV penalty only when recon ≤ budget × warmup_recon
              "maximize CV subject to reconstruction staying good enough"

Usage:
    cd Fig6
    python compare_options.py             # train all + evaluate + plot
    python compare_options.py --eval-only # skip training
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as scio
import torch
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(BASE_DIR, 'record')
DATA_PATH  = os.path.join(BASE_DIR, 'src', 'model', 'IMAGES.mat')

N_RUNS   = 2    # quick scan; use 5+ for paper-quality
N_EPOCHS = 10   # match USE_NEW checkpoint convention

# ── Conditions to test ─────────────────────────────────────────────────────────
# Each dict: cons, cv_ratio, reg (lmda), recon_budget, label, color
CONDITIONS = [
    # ── Option 1 best: current default ──────────────────────────────────────
    dict(cons='cv',            cv_ratio=0.10, reg=5e-3, recon_budget=1.0,
         label='Opt1  cv=0.10 λ=5e-3',         color='#E64B35', marker='o'),
    dict(cons='cv',            cv_ratio=0.04, reg=5e-3, recon_budget=1.0,
         label='Opt1  cv=0.04 λ=5e-3',         color='#F39B7F', marker='o'),

    # ── Option 2: lower lmda (soft threshold) ────────────────────────────────
    dict(cons='cv',            cv_ratio=0.05, reg=1e-3, recon_budget=1.0,
         label='Opt2  cv=0.05 λ=1e-3',         color='#4DBBD5', marker='s'),
    dict(cons='cv',            cv_ratio=0.08, reg=1e-3, recon_budget=1.0,
         label='Opt2  cv=0.08 λ=1e-3',         color='#3C5488', marker='s'),
    dict(cons='cv',            cv_ratio=0.10, reg=1e-3, recon_budget=1.0,
         label='Opt2  cv=0.10 λ=1e-3',         color='#00A087', marker='s'),

    # ── Option 3: constrained (recon budget) ─────────────────────────────────
    dict(cons='cv_constrained', cv_ratio=0.10, reg=5e-3, recon_budget=1.2,
         label='Opt3  budget=1.2 (lenient)',    color='#7E6148', marker='^'),
    dict(cons='cv_constrained', cv_ratio=0.10, reg=5e-3, recon_budget=1.0,
         label='Opt3  budget=1.0 (tight)',      color='#B09C85', marker='^'),
    dict(cons='cv_constrained', cv_ratio=0.10, reg=1e-3, recon_budget=1.2,
         label='Opt3  cv=0.10 λ=1e-3 bgt=1.2', color='#8491B4', marker='^'),
]

# ── Baselines (from result_new.py output) ──────────────────────────────────────
BASELINES = {
    'null':   dict(RE=0.6110, CV=2.629, cost=0.0578, ICE=12.300),
    'cauchy': dict(RE=0.7020, CV=2.480, cost=0.0419, ICE=12.175),
}


# ── Tag helpers ────────────────────────────────────────────────────────────────
def make_tag(cond: dict) -> str:
    """Unique filesystem-safe tag for a condition."""
    c = cond['cons'].replace('_constrained', 'con')
    return f"{c}_cv{cond['cv_ratio']:.2f}_lm{cond['reg']:.0e}_bgt{cond['recon_budget']:.1f}"


def make_suffix(cond: dict) -> str:
    name_map = {'non': '_16_new', 's': '_cauchy_16_nl_new',
                'cv': '_cv_16_nl_new', 'cv_constrained': '_cv_con_new'}
    base = name_map[cond['cons']]
    return base + '_' + make_tag(cond)


# ── Training ───────────────────────────────────────────────────────────────────
def run_training(conditions, n_runs, n_epochs):
    for cond in conditions:
        t = make_tag(cond)
        print(f"\n{'='*60}\nTraining: {cond['label']}  (tag={t})\n{'='*60}")
        cmd = [
            sys.executable, os.path.join(BASE_DIR, 'train_new.py'),
            '--cons',        cond['cons'],
            '--cv_ratio',    str(cond['cv_ratio']),
            '--reg',         str(cond['reg']),
            '--recon-budget',str(cond['recon_budget']),
            '--n_runs',      str(n_runs),
            '--run_start',   '0',
            '--epoch',       str(n_epochs),
            '--tag',         t,
        ]
        subprocess.run(cmd, check=True, cwd=BASE_DIR)


# ── Evaluation (shared with pareto_scan) ───────────────────────────────────────
def evaluate_condition(cond, n_runs, n_epochs, X, img_size, n_img):
    sfx       = make_suffix(cond)
    model_dir = os.path.join(RECORD_DIR, 'trained_models' + sfx)

    re_list, cv_list, cost_list = [], [], []

    for k in range(n_runs):
        rec_path = os.path.join(RECORD_DIR, f'record{sfx}_{k}.mat')
        if not os.path.exists(rec_path):
            print(f"  [SKIP] {rec_path}")
            continue
        record = scio.loadmat(rec_path)

        re_acc = cost_acc = cv_acc = 0.0
        n_loaded = 0

        for j in range(n_img):
            img = X[:, :, j]
            pth = os.path.join(model_dir, f'ckpt-{n_epochs}-{k}-{j+1}.pth')
            if not os.path.exists(pth):
                continue

            net = torch.load(pth, map_location='cpu')
            w   = torch.abs(net.R).detach().numpy()

            cost_acc += np.mean(w)
            cv_pp = np.mean([np.std(w[i]) / (np.mean(w[i]) + 1e-12) for i in range(w.shape[0])])
            cv_acc += cv_pp

            recon   = net.U(net.R).detach().numpy()
            pw      = int(np.sqrt(recon.shape[1]))
            img_rec = np.zeros([img_size, img_size])
            ppr     = img_size // pw
            for i in range(recon.shape[0]):
                x0, y0 = (i % ppr) * pw, (i // ppr) * pw
                if x0 + pw <= img_size and y0 + pw <= img_size:
                    img_rec[x0:x0+pw, y0:y0+pw] = recon[i].reshape(pw, pw)

            re_acc += np.linalg.norm(img - img_rec) / (np.linalg.norm(img) + 1e-12)
            n_loaded += 1

        if n_loaded == 0:
            continue
        re_list.append(re_acc / n_loaded)
        cv_list.append(cv_acc / n_loaded)
        cost_list.append(cost_acc / n_loaded)

    if not re_list:
        return None

    re_m, re_s     = np.mean(re_list),   np.std(re_list)
    cv_m, cv_s     = np.mean(cv_list),   np.std(cv_list)
    cost_m         = np.mean(cost_list)
    ice            = -np.log2(re_m + 1e-12) / (cost_m + 1e-12)
    print(f"  {cond['label']:45s}  RE={re_m:.4f}±{re_s:.4f}  "
          f"CV={cv_m:.3f}  ā={cost_m:.4f}  ICE={ice:.2f}")
    return dict(**cond, RE=re_m, RE_std=re_s, CV=cv_m, CV_std=cv_s,
                cost=cost_m, ICE=ice)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_comparison(results):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)

    # Reference lines from baselines
    for ax in axes:
        ax.axhline(BASELINES['null']['RE'],   ls='--', lw=1, color='gray',
                   alpha=0.7, label='null RE')
        ax.axhline(BASELINES['cauchy']['RE'], ls='--', lw=1, color='#91D1C2',
                   alpha=0.9, label='cauchy RE')

    # ── Panel 1: RE vs CV ──────────────────────────────────────────────────
    ax = axes[0]
    for r in results:
        ax.errorbar(r['CV'], r['RE'], xerr=r['CV_std'], yerr=r['RE_std'],
                    fmt=r['marker'], color=r['color'], markersize=8,
                    label=r['label'], capsize=3)
    for name, b in BASELINES.items():
        ax.scatter(b['CV'], b['RE'], marker='D', s=80, color='gray',
                   zorder=5, label=name)
    ax.set_xlabel('Population CV', fontsize=13)
    ax.set_ylabel('Relative reconstruction error (RE)', fontsize=13)
    ax.set_title('RE vs CV', fontsize=13)

    # Panel 2: RAE vs CV (internal arrays retain the legacy ICE key).
    ax = axes[1]
    for b in BASELINES.values():
        ax.axhline(b['ICE'], ls='--', lw=1, color='gray', alpha=0.6)
    for r in results:
        ax.scatter(r['CV'], r['ICE'], marker=r['marker'],
                   color=r['color'], s=80, label=r['label'])
    ax.set_xlabel('Population CV', fontsize=13)
    ax.set_ylabel(r'RAE score $= -\log_2(\mathrm{RE})/\bar{a}$', fontsize=11)
    ax.set_title('RAE vs CV', fontsize=13)

    # ── Panel 3: RE vs ā (cost) ───────────────────────────────────────────
    ax = axes[2]
    for r in results:
        ax.scatter(r['cost'], r['RE'], marker=r['marker'],
                   color=r['color'], s=80, label=r['label'])
    for name, b in BASELINES.items():
        ax.scatter(b['cost'], b['RE'], marker='D', s=80, color='gray',
                   zorder=5, label=name)
    ax.set_xlabel(r'Mean activation $\bar{a}$ (cost)', fontsize=13)
    ax.set_ylabel('Relative reconstruction error (RE)', fontsize=13)
    ax.set_title(r'RE vs $\bar{a}$', fontsize=13)

    # Shared legend (rightmost panel)
    handles = ([plt.Line2D([0], [0], marker=r['marker'], color='w',
                            markerfacecolor=r['color'], markersize=8,
                            label=r['label']) for r in results]
               + [mpatches.Patch(color='gray', alpha=0.5, label='null / cauchy')])
    axes[2].legend(handles=handles, fontsize=7.5, frameon=False,
                   bbox_to_anchor=(1.02, 1), loc='upper left')

    fig.tight_layout()
    out = os.path.join(BASE_DIR, 'compare_options.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f"\nSaved → {out}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--n-runs',   type=int, default=N_RUNS)
    p.add_argument('--n-epochs', type=int, default=N_EPOCHS)
    args = p.parse_args()

    if not args.eval_only:
        run_training(CONDITIONS, args.n_runs, args.n_epochs)

    print("\nEvaluating all conditions...")
    X      = loadmat(DATA_PATH)['IMAGES']
    img_sz = X.shape[0]
    n_img  = X.shape[2]

    results = []
    for cond in CONDITIONS:
        r = evaluate_condition(cond, args.n_runs, args.n_epochs, X, img_sz, n_img)
        if r is not None:
            results.append(r)

    if not results:
        print("No results found. Run without --eval-only first.")
        sys.exit(1)

    df = pd.DataFrame(results)
    print("\n── Summary ──────────────────────────────────────────────────────────────")
    cols = ['label', 'RE', 'CV', 'cost', 'ICE']
    print(df[cols].to_string(index=False, float_format='{:.4f}'.format))

    plot_comparison(results)
