"""
Generate the supp figure that retains the original Fig 2 narrative based on
POPULATION CV (CV_P, per-trial across-neuron heterogeneity).  This is the
'opposite-projection' check: CV_P probes within-trial across-neuron
variability (dominant in evoked due to stim tuning), whereas main-text
Fig 2 (CV_L) probes across-trial single-neuron variability (dominant in
spontaneous due to temporal drift).

Each dot = one trial; spontaneous (blue) and evoked (orange) clusters.

Output: FigS_CVP_supp.pdf
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
OUT_PDF = HERE / "FigS_CVP_supp.pdf"

N_V1       = 159
DUR_SPON   = 1.5
DUR_DRIVEN = 1.28


def vinje_gallant_S_per_trial(R):
    """S_P per trial across neurons; R is (N_neurons, n_trials)."""
    mu  = R.mean(axis=0)
    sec = (R ** 2).mean(axis=0)
    n   = R.shape[0]
    return (1.0 - mu ** 2 / sec) / (1.0 - 1.0 / n)


def main():
    m = sio.loadmat(str(MAT), squeeze_me=False, struct_as_record=False)
    nd = m["neuralData"][0, 0]
    rasters = nd.spikeRasters

    spon_rates, evk_rates = [], []
    for i in range(rasters.shape[0]):
        r = rasters[i, 0]
        if r.shape[1] == 1500:
            spon_rates.append(np.asarray(r.sum(axis=1)).ravel() / DUR_SPON)
        elif r.shape[1] == 1280:
            evk_rates.append(np.asarray(r.sum(axis=1)).ravel() / DUR_DRIVEN)
    spon = np.array(spon_rates).T
    evk  = np.array(evk_rates).T

    # Per-trial population stats
    mu_spon = spon.mean(axis=0)
    cv_spon = spon.std(axis=0) / mu_spon
    s_spon  = vinje_gallant_S_per_trial(spon)
    mu_evk  = evk.mean(axis=0)
    cv_evk  = evk.std(axis=0) / mu_evk
    s_evk   = vinje_gallant_S_per_trial(evk)

    print("=== per-trial population CV_P / S_P (V1) ===")
    print(f"           spon median   evk median")
    print(f"  mean_FR  {np.median(mu_spon):8.3f}   {np.median(mu_evk):8.3f}  Hz")
    print(f"  CV_P     {np.median(cv_spon):8.4f}   {np.median(cv_evk):8.4f}")
    print(f"  S_P      {np.median(s_spon):8.4f}   {np.median(s_evk):8.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    blue   = (0.20, 0.45, 0.85)
    orange = (1.00, 0.55, 0.20)

    axA = axes[0]
    axA.scatter(mu_spon, s_spon, s=10, color=blue,   alpha=0.35, edgecolor='none', label='Spontaneous (per trial)')
    axA.scatter(mu_evk,  s_evk,  s=10, color=orange, alpha=0.35, edgecolor='none', label='Evoked (per trial)')
    axA.set_xlabel(r'Per-trial mean firing rate $\bar{r}$ (Hz)')
    axA.set_ylabel(r'Population sparseness $S_P$')
    axA.set_title(r'(A)  $S_P$ vs $\bar{r}$ (per trial)')
    axA.legend(frameon=False, loc='lower right', fontsize=9)

    axB = axes[1]
    axB.scatter(mu_spon, cv_spon, s=10, color=blue,   alpha=0.35, edgecolor='none', label='Spontaneous (per trial)')
    axB.scatter(mu_evk,  cv_evk,  s=10, color=orange, alpha=0.35, edgecolor='none', label='Evoked (per trial)')
    axB.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.6)
    axB.set_xlabel(r'Per-trial mean firing rate $\bar{r}$ (Hz)')
    axB.set_ylabel(r'Population CV  $\mathrm{CV}_P$')
    axB.set_title(r'(B)  $\mathrm{CV}_P$ vs $\bar{r}$ (per trial)')
    axB.legend(frameon=False, loc='lower right', fontsize=9)

    axC = axes[2]
    axC.scatter(cv_spon, s_spon, s=10, color=blue,   alpha=0.35, edgecolor='none', label='Spontaneous')
    axC.scatter(cv_evk,  s_evk,  s=10, color=orange, alpha=0.35, edgecolor='none', label='Evoked')
    cv_grid = np.linspace(0.05, 2.5, 400)
    axC.plot(cv_grid, cv_grid ** 2 / (1 + cv_grid ** 2), 'k-', lw=1.5,
             label=r'$S = \mathrm{CV}^2/(1+\mathrm{CV}^2)$')
    axC.set_xlabel(r'Population CV  $\mathrm{CV}_P$')
    axC.set_ylabel(r'Population sparseness  $S_P$')
    axC.set_title(r'(C)  $S_P$ vs $\mathrm{CV}_P$ (algebraic identity)')
    axC.legend(frameon=False, loc='lower right', fontsize=9)
    axC.set_xlim(0, max(cv_spon.max(), cv_evk.max()) * 1.05)
    axC.set_ylim(0, 1.0)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Supp.  V1 population statistics (CV_P, S_P, per trial across neurons): ' \
                 'evoked > spontaneous in CV_P/S_P with very small effect size; ' \
                 'main spontaneous--evoked separation is in firing rate.',
                 fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    print(f"[save] {OUT_PDF}")


if __name__ == "__main__":
    main()
