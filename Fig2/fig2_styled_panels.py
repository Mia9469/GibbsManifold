"""
Generate the styled C + D panels matching the reference layout:
- Panel C left:  S  vs mean firing rate (scatter; spon blue / evk orange) + inset bar chart with significance star
- Panel C right: CV vs mean firing rate (same style + inset)
- Panel D:       S vs CV scatter + theoretical curve S = CV^2/(1+CV^2)
Two figures are produced:
  Fig2_panels_CD_CVL_styled.pdf   --- per-neuron lifetime CV/S (main-text Fig 2)
  FigS_CVP_styled.pdf             --- per-trial population CV/S (supp Fig S3)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.family']  = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size']    = 10
mpl.rcParams['axes.linewidth'] = 0.9

HERE = Path(__file__).resolve().parent
MAT  = HERE / "105l001p16.mat"

BLUE   = (0.20, 0.45, 0.85)
ORANGE = (1.00, 0.55, 0.20)

MATCHED_WINDOW_S = 1.28
DUR_SPON   = MATCHED_WINDOW_S
DUR_DRIVEN = MATCHED_WINDOW_S
N_BINS_MATCHED = 1280
N_V1 = 159


def vinje_gallant_S(R, axis):
    n = R.shape[axis]
    mu = R.mean(axis=axis)
    sec = (R**2).mean(axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 - mu**2 / sec) / (1.0 - 1.0 / n)


def load_rates():
    m = sio.loadmat(str(MAT), squeeze_me=False, struct_as_record=False)
    nd = m["neuralData"][0, 0]
    rasters = nd.spikeRasters
    spon, evk = [], []
    for i in range(rasters.shape[0]):
        r = rasters[i, 0]
        if r.shape[1] == 1500:
            spon.append(np.asarray(r[:, :N_BINS_MATCHED].sum(axis=1)).ravel() / DUR_SPON)
        elif r.shape[1] == 1280:
            evk.append(np.asarray(r.sum(axis=1)).ravel() / DUR_DRIVEN)
    return np.array(spon).T, np.array(evk).T   # both (159, n_trials)


def stars(p):
    if p < 1e-4: return '****'
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 5e-2: return '*'
    return 'ns'


def add_inset_bars(ax, spon_vals, evk_vals, ylabel, p_value,
                   loc='upper right', width="40%", height="55%"):
    """Inset bar chart: medians +/- IQR for spon vs evk, with significance star."""
    axin = inset_axes(ax, width=width, height=height, loc=loc,
                      borderpad=0.6)
    meds   = [np.median(spon_vals), np.median(evk_vals)]
    iqrs_l = [meds[0] - np.quantile(spon_vals, 0.25),
              meds[1] - np.quantile(evk_vals,  0.25)]
    iqrs_h = [np.quantile(spon_vals, 0.75) - meds[0],
              np.quantile(evk_vals,  0.75) - meds[1]]
    xs = [0, 1]
    axin.bar(xs, meds, color=[BLUE, ORANGE], width=0.6, alpha=0.85,
             edgecolor='black', linewidth=0.6)
    axin.errorbar(xs, meds, yerr=[iqrs_l, iqrs_h], fmt='none',
                  ecolor='black', capsize=3, lw=0.8)
    # significance bar
    y_top = max(meds[0]+iqrs_h[0], meds[1]+iqrs_h[1])
    span  = y_top - min(meds)
    y_bar = y_top + 0.10 * span
    axin.plot([0, 0, 1, 1], [y_bar, y_bar*1.02, y_bar*1.02, y_bar],
              lw=0.7, color='black')
    axin.text(0.5, y_bar*1.04, stars(p_value),
              ha='center', va='bottom', fontsize=9)
    axin.set_xticks(xs); axin.set_xticklabels(['spon','evk'], fontsize=7)
    axin.set_ylabel(ylabel, fontsize=8)
    axin.tick_params(axis='y', labelsize=7)
    # tight ylim around the bars
    y_lo = max(0, min(meds) - max(iqrs_l) - 0.05*span)
    y_hi = y_bar*1.10
    axin.set_ylim(y_lo, y_hi)
    axin.set_xlim(-0.6, 1.6)
    for spine in ['top', 'right']:
        axin.spines[spine].set_visible(False)


def render_panels(mu_spon, S_spon, CV_spon, mu_evk, S_evk, CV_evk,
                  out_pdf, label_S, label_CV, paired,
                  s_ylim, cv_ylim, fr_xlim, identity_xlim):
    """Generic panel renderer used for both CV_L and CV_P versions."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4),
                             gridspec_kw={'width_ratios': [1.0, 1.0, 1.0]})

    # significance tests
    if paired:
        _, p_S  = wilcoxon(S_spon, S_evk, alternative='two-sided')
        _, p_CV = wilcoxon(CV_spon, CV_evk, alternative='two-sided')
    else:
        _, p_S  = mannwhitneyu(S_spon, S_evk, alternative='two-sided')
        _, p_CV = mannwhitneyu(CV_spon, CV_evk, alternative='two-sided')

    print(f"  {label_S}:  p={p_S:.2e}  ({stars(p_S)})    "
          f"{label_CV}: p={p_CV:.2e}  ({stars(p_CV)})")

    # --- C left: S vs FR ---
    axCL = axes[0]
    axCL.scatter(mu_spon, S_spon, s=12, color=BLUE,   alpha=0.7, edgecolor='none')
    axCL.scatter(mu_evk,  S_evk,  s=12, color=ORANGE, alpha=0.7, edgecolor='none')
    axCL.set_xlabel(r'$\bar{f}$ (Hz)', fontsize=11)
    axCL.set_ylabel(label_S, fontsize=12)
    if s_ylim:  axCL.set_ylim(*s_ylim)
    if fr_xlim: axCL.set_xlim(*fr_xlim)
    add_inset_bars(axCL, S_spon, S_evk, label_S, p_S, loc='upper right')

    # --- C right: CV vs FR ---
    axCR = axes[1]
    axCR.scatter(mu_spon, CV_spon, s=12, color=BLUE,   alpha=0.7, edgecolor='none')
    axCR.scatter(mu_evk,  CV_evk,  s=12, color=ORANGE, alpha=0.7, edgecolor='none')
    axCR.set_xlabel(r'$\bar{f}$ (Hz)', fontsize=11)
    axCR.set_ylabel(label_CV, fontsize=12)
    if cv_ylim: axCR.set_ylim(*cv_ylim)
    if fr_xlim: axCR.set_xlim(*fr_xlim)
    add_inset_bars(axCR, CV_spon, CV_evk, label_CV, p_CV, loc='upper right')

    # --- D: S vs CV identity ---
    axD = axes[2]
    axD.scatter(CV_spon, S_spon, s=14, color=BLUE,   alpha=0.7, edgecolor='none')
    axD.scatter(CV_evk,  S_evk,  s=14, color=ORANGE, alpha=0.7, edgecolor='none')
    cv_grid = np.linspace(0.0, identity_xlim[1], 400)
    axD.plot(cv_grid, cv_grid**2 / (1 + cv_grid**2), 'k-', lw=1.2)
    axD.set_xlabel(label_CV, fontsize=12)
    axD.set_ylabel(label_S, fontsize=12)
    axD.set_xlim(*identity_xlim); axD.set_ylim(0, 1.0)
    axD.text(0.04, 0.94, 'Experiment\nData', transform=axD.transAxes,
             ha='left', va='top', fontsize=9)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=9)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"[save] {out_pdf}")


# --- main ---

print("[load]")
spon, evk = load_rates()
print(f"  spon {spon.shape}, evk {evk.shape}")


# === Per-neuron LIFETIME CV/S (main-text Fig 2) ===
mu_spon_L  = spon.mean(axis=1)
sd_spon_L  = spon.std(axis=1)
cvL_spon   = sd_spon_L / np.where(mu_spon_L > 1e-9, mu_spon_L, np.nan)
sL_spon    = vinje_gallant_S(spon, axis=1)

mu_evk_L  = evk.mean(axis=1)
sd_evk_L  = evk.std(axis=1)
cvL_evk   = sd_evk_L / np.where(mu_evk_L > 1e-9, mu_evk_L, np.nan)
sL_evk    = vinje_gallant_S(evk, axis=1)

valid = (mu_spon_L > 1e-3) & (mu_evk_L > 1e-3) & np.isfinite(cvL_spon) & np.isfinite(cvL_evk)
print(f"\n[CV_L] {valid.sum()} neurons used")
render_panels(mu_spon_L[valid], sL_spon[valid], cvL_spon[valid],
              mu_evk_L[valid],  sL_evk[valid],  cvL_evk[valid],
              out_pdf=HERE / "Fig2_panels_CD_CVL_styled.pdf",
              label_S=r'$S_L$', label_CV=r'$\mathrm{CV}_L$',
              paired=True,
              s_ylim=(0, 1.0), cv_ylim=(0, 2.2),
              fr_xlim=(0, 30),
              identity_xlim=(0, 2.5))


# === Per-trial POPULATION CV/S (supp Fig S3) ===
mu_spon_P = spon.mean(axis=0)
cvP_spon  = spon.std(axis=0) / mu_spon_P
sP_spon   = vinje_gallant_S(spon, axis=0)

mu_evk_P  = evk.mean(axis=0)
cvP_evk   = evk.std(axis=0) / mu_evk_P
sP_evk    = vinje_gallant_S(evk, axis=0)

print(f"\n[CV_P] {len(mu_spon_P)} spon trials, {len(mu_evk_P)} evk trials")
render_panels(mu_spon_P, sP_spon, cvP_spon,
              mu_evk_P,  sP_evk,  cvP_evk,
              out_pdf=HERE / "FigS_CVP_styled.pdf",
              label_S=r'$S_P$', label_CV=r'$\mathrm{CV}_P$',
              paired=False,
              s_ylim=(0.2, 1.0), cv_ylim=(0.5, 2.0),
              fr_xlim=(0, 22),
              identity_xlim=(0, 3.0))
