"""
Generate Fig 2 panels C and D using LIFETIME CV (CV_L) and lifetime
sparseness (S_L), per-neuron, separately for spontaneous and evoked.
Each dot in panels C and D is one neuron; colours separate condition.

Panel C (left)  : S_L  vs per-neuron mean firing rate
Panel C (right) : CV_L vs per-neuron mean firing rate
Panel D         : S_L  vs CV_L scatter + theoretical S = CV^2 / (1+CV^2)

Per the new narrative: CV_L (per-neuron, across-trial variability) is
the metric that distinguishes V1 spontaneous (high CV_L) from evoked
(lower CV_L), since stimulus-locking constrains single-neuron rate
across trials.

Output:
  Fig2_panels_CD_CVL.pdf   (main-text replacement)
  Fig2_panels_CD_CVL_data.npz (raw arrays, for downstream use)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.family']  = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size']    = 10

HERE = Path(__file__).resolve().parent
MAT  = HERE / "105l001p16.mat"
OUT_PDF = HERE / "Fig2_panels_CD_CVL.pdf"
OUT_NPZ = HERE / "Fig2_panels_CD_CVL_data.npz"

N_V1       = 159
MATCHED_WINDOW_S = 1.28
DUR_SPON   = MATCHED_WINDOW_S
DUR_DRIVEN = MATCHED_WINDOW_S
N_BINS_MATCHED = 1280


def vinje_gallant_S(rates_along_axis, axis):
    """Vinje-Gallant bias-corrected sparseness along the given axis.
    S = (1 - <r>^2 / <r^2>) / (1 - 1/N)
    where N is the number of samples along that axis."""
    n = rates_along_axis.shape[axis]
    mu  = rates_along_axis.mean(axis=axis)
    sec = (rates_along_axis ** 2).mean(axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (1.0 - mu ** 2 / sec) / (1.0 - 1.0 / n)
    return s


def main():
    print(f"[load] {MAT}")
    m = sio.loadmat(str(MAT), squeeze_me=False, struct_as_record=False)
    nd = m["neuralData"][0, 0]
    rasters = nd.spikeRasters
    n_trials_total = rasters.shape[0]

    spon_rates, evk_rates = [], []
    for i in range(n_trials_total):
        r = rasters[i, 0]
        if r.shape[1] == 1500:
            # Match the spontaneous integration window to the 1.28 s
            # driven window; count CV depends on window duration.
            spon_rates.append(np.asarray(r[:, :N_BINS_MATCHED].sum(axis=1)).ravel() / DUR_SPON)
        elif r.shape[1] == 1280:
            evk_rates.append(np.asarray(r.sum(axis=1)).ravel() / DUR_DRIVEN)
    spon = np.array(spon_rates).T   # (159, n_spon_trials)
    evk  = np.array(evk_rates).T    # (159, n_evk_trials)
    print(f"[shape] spon {spon.shape}, evk {evk.shape}")

    # Per-neuron lifetime stats: aggregate across trials within each condition.
    mu_spon  = spon.mean(axis=1)                              # (159,)
    sd_spon  = spon.std(axis=1)
    cvL_spon = sd_spon / np.where(mu_spon > 1e-9, mu_spon, np.nan)
    sL_spon  = vinje_gallant_S(spon, axis=1)

    mu_evk  = evk.mean(axis=1)
    sd_evk  = evk.std(axis=1)
    cvL_evk = sd_evk / np.where(mu_evk > 1e-9, mu_evk, np.nan)
    sL_evk  = vinje_gallant_S(evk, axis=1)

    # Drop neurons with mean rate <= 0 in either condition to avoid undefined CV
    valid = (mu_spon > 1e-3) & (mu_evk > 1e-3)
    print(f"[filter] kept {valid.sum()} / {N_V1} neurons after FR>0 in both conditions")
    mu_spon_v, cvL_spon_v, sL_spon_v = mu_spon[valid], cvL_spon[valid], sL_spon[valid]
    mu_evk_v,  cvL_evk_v,  sL_evk_v  = mu_evk[valid],  cvL_evk[valid],  sL_evk[valid]

    # --- summary ---
    print("\n=== per-neuron lifetime stats (V1) ===")
    print(f"{'':>12s}  {'spon (med)':>12s}  {'evk (med)':>12s}")
    print(f"{'mean FR Hz':>12s}  {np.median(mu_spon_v):12.3f}  {np.median(mu_evk_v):12.3f}")
    print(f"{'CV_L':>12s}  {np.median(cvL_spon_v):12.4f}  {np.median(cvL_evk_v):12.4f}")
    print(f"{'S_L (V-G)':>12s}  {np.median(sL_spon_v):12.4f}  {np.median(sL_evk_v):12.4f}")

    # --- save data for reuse ---
    np.savez(OUT_NPZ,
             mu_spon=mu_spon_v, cvL_spon=cvL_spon_v, sL_spon=sL_spon_v,
             mu_evk=mu_evk_v,   cvL_evk=cvL_evk_v,   sL_evk=sL_evk_v,
             spon_window_s=MATCHED_WINDOW_S, evk_window_s=MATCHED_WINDOW_S)
    print(f"[save] {OUT_NPZ}")

    # --- plot panels C and D ---
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    blue   = (0.20, 0.45, 0.85)
    orange = (1.00, 0.55, 0.20)

    # Panel C-left: S_L vs FR
    axC1 = axes[0]
    axC1.scatter(mu_spon_v, sL_spon_v, s=24, color=blue,   alpha=0.7,
                 edgecolor='none', label='Spontaneous')
    axC1.scatter(mu_evk_v,  sL_evk_v,  s=24, color=orange, alpha=0.7,
                 edgecolor='none', label='Evoked')
    axC1.set_xlabel(r'Mean firing rate $\bar{r}$ (Hz)')
    axC1.set_ylabel(r'Lifetime sparseness $S_L$')
    axC1.set_title(r'(C, left)  $S_L$  vs  $\bar{r}$')
    axC1.legend(frameon=False, loc='lower right', fontsize=9)
    axC1.set_xscale('log')

    # Panel C-right: CV_L vs FR
    axC2 = axes[1]
    axC2.scatter(mu_spon_v, cvL_spon_v, s=24, color=blue,   alpha=0.7,
                 edgecolor='none', label='Spontaneous')
    axC2.scatter(mu_evk_v,  cvL_evk_v,  s=24, color=orange, alpha=0.7,
                 edgecolor='none', label='Evoked')
    axC2.set_xlabel(r'Mean firing rate $\bar{r}$ (Hz)')
    axC2.set_ylabel(r'Lifetime CV  $\mathrm{CV}_L$')
    axC2.set_title(r'(C, right)  $\mathrm{CV}_L$  vs  $\bar{r}$')
    axC2.legend(frameon=False, loc='upper right', fontsize=9)
    axC2.set_xscale('log')

    # Panel D: S_L vs CV_L identity curve + ISI sim placeholder note
    axD = axes[2]
    axD.scatter(cvL_spon_v, sL_spon_v, s=24, color=blue,   alpha=0.7,
                edgecolor='none', label='Spontaneous')
    axD.scatter(cvL_evk_v,  sL_evk_v,  s=24, color=orange, alpha=0.7,
                edgecolor='none', label='Evoked')
    cv_grid = np.linspace(0.05, 2.5, 400)
    axD.plot(cv_grid, cv_grid ** 2 / (1 + cv_grid ** 2), 'k-', lw=1.5,
             label=r'$S = \mathrm{CV}^2/(1+\mathrm{CV}^2)$')
    axD.set_xlabel(r'Lifetime CV  $\mathrm{CV}_L$')
    axD.set_ylabel(r'Lifetime sparseness  $S_L$')
    axD.set_title(r'(D)  $S_L$ vs $\mathrm{CV}_L$ (algebraic identity)')
    axD.legend(frameon=False, loc='lower right', fontsize=9)
    axD.set_xlim(0, max(cvL_spon_v.max(), cvL_evk_v.max()) * 1.05)
    axD.set_ylim(0, 1.0)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Fig 2 panels C, D --- replotted with lifetime statistics ' \
                 '(per-neuron CV_L and S_L across trials)',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    print(f"[save] {OUT_PDF}")


if __name__ == "__main__":
    main()
