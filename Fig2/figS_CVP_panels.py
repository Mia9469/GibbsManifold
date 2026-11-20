"""Generate Supplementary Fig. S3: pooled V1 population CV_P/S_P checks.

The original S3 used only session 105l001p16, where the per-trial
population projection is slightly evoked-higher.  This updated version
uses all five v1v2-1 sessions used elsewhere in the manuscript and makes
the estimator-specific heterogeneity explicit:

  - CV_L remains the main Fig. 2 axis because it samples across-trial
    single-neuron variability.
  - CV_P samples across-neuron heterogeneity within each trial/window.
    This projection is weaker in V1 and has one single-session exception,
    but 4/5 sessions show spontaneous > evoked in session-median CV_P.

Outputs
-------
Fig2/FigureS3.pdf
Fig2/FigureS3.png
Fig2/FigS_CVP_supp.pdf          (legacy alias)
Fig2/figS_CVP_summary.json

The FigureS3 PDF/PNG are copied into updated/figures/ when that directory
exists.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
UPDATED_FIGURES = REPO / "updated" / "figures"

OUT_PDF = HERE / "FigureS3.pdf"
OUT_PNG = HERE / "FigureS3.png"
OUT_LEGACY = HERE / "FigS_CVP_supp.pdf"
OUT_JSON = HERE / "figS_CVP_summary.json"

V1V2_FILES = [
    "105l001p16.mat",
    "106r001p26.mat",
    "106r002p70.mat",
    "107l002p67.mat",
    "107l003p143.mat",
]

MATCHED_WINDOW_S = 1.28
DUR_SPON = MATCHED_WINDOW_S
DUR_DRIVEN = MATCHED_WINDOW_S
N_BINS_MATCHED = 1280
BLUE = "#0072bd"
ORANGE = "#d95319"
EXCEPTION = "#C95F5F"
CVP_MARKER = "o"
SP_MARKER = "s"


def vinje_gallant_s_population(rates: np.ndarray) -> np.ndarray:
    """Per-trial population sparseness across neurons.

    rates has shape (n_neurons, n_trials).
    """
    mu = rates.mean(axis=0)
    sec = (rates**2).mean(axis=0)
    n = rates.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 - mu**2 / sec) / (1.0 - 1.0 / n)


def load_rates(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    rasters = m["neuralData"][0, 0].spikeRasters
    spon = []
    evk = []
    for i in range(rasters.shape[0]):
        r = rasters[i, 0]
        if r.shape[1] == 1500:
            spon.append(np.asarray(r[:, :N_BINS_MATCHED].sum(axis=1)).ravel() / DUR_SPON)
        elif r.shape[1] == 1280:
            evk.append(np.asarray(r.sum(axis=1)).ravel() / DUR_DRIVEN)
    return np.array(spon).T, np.array(evk).T


def population_metrics(rates: np.ndarray) -> dict[str, np.ndarray]:
    mu = rates.mean(axis=0)
    cv = rates.std(axis=0) / np.where(mu > 1e-12, mu, np.nan)
    sp = vinje_gallant_s_population(rates)
    return {"fr": mu, "cvp": cv, "sp": sp}


def load_all_sessions() -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = []
    pooled = {
        "fr_spon": [],
        "cvp_spon": [],
        "sp_spon": [],
        "fr_evk": [],
        "cvp_evk": [],
        "sp_evk": [],
        "session_spon": [],
        "session_evk": [],
    }

    for session_i, filename in enumerate(V1V2_FILES):
        spon_rates, evk_rates = load_rates(HERE / filename)
        spon = population_metrics(spon_rates)
        evk = population_metrics(evk_rates)

        row = {
            "session": filename.replace(".mat", ""),
            "n_units": int(spon_rates.shape[0]),
            "n_spontaneous_trials": int(spon_rates.shape[1]),
            "n_evoked_trials": int(evk_rates.shape[1]),
            "spontaneous_median_FR_Hz": float(np.nanmedian(spon["fr"])),
            "evoked_median_FR_Hz": float(np.nanmedian(evk["fr"])),
            "spontaneous_median_CV_P": float(np.nanmedian(spon["cvp"])),
            "evoked_median_CV_P": float(np.nanmedian(evk["cvp"])),
            "delta_CV_P_spont_minus_evoked": float(
                np.nanmedian(spon["cvp"]) - np.nanmedian(evk["cvp"])
            ),
            "spontaneous_median_S_P": float(np.nanmedian(spon["sp"])),
            "evoked_median_S_P": float(np.nanmedian(evk["sp"])),
            "delta_S_P_spont_minus_evoked": float(
                np.nanmedian(spon["sp"]) - np.nanmedian(evk["sp"])
            ),
            "mannwhitney_CV_P_p_two_sided": float(
                mannwhitneyu(spon["cvp"], evk["cvp"], alternative="two-sided").pvalue
            ),
            "mannwhitney_S_P_p_two_sided": float(
                mannwhitneyu(spon["sp"], evk["sp"], alternative="two-sided").pvalue
            ),
        }
        rows.append(row)

        for key in ("fr", "cvp", "sp"):
            pooled[f"{key}_spon"].append(spon[key])
            pooled[f"{key}_evk"].append(evk[key])
        pooled["session_spon"].append(np.full_like(spon["cvp"], session_i, dtype=int))
        pooled["session_evk"].append(np.full_like(evk["cvp"], session_i, dtype=int))

    pooled_arrays = {k: np.concatenate(v) for k, v in pooled.items()}
    return rows, pooled_arrays


def boot_median_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Median and percentile bootstrap 95% CI of the median."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    n = len(x)
    for b in range(n_boot):
        meds[b] = np.median(x[rng.integers(0, n, n)])
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def p_label(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def summarize(rows: list[dict], pooled: dict[str, np.ndarray]) -> dict:
    dcvp = np.array([r["delta_CV_P_spont_minus_evoked"] for r in rows], dtype=float)
    dsp = np.array([r["delta_S_P_spont_minus_evoked"] for r in rows], dtype=float)
    dfr = np.array([r["spontaneous_median_FR_Hz"] - r["evoked_median_FR_Hz"] for r in rows], dtype=float)
    out = {
        "per_session": rows,
        "n_sessions": len(rows),
        "pooled_trials_per_condition": {
            "spontaneous": int(len(pooled["cvp_spon"])),
            "evoked": int(len(pooled["cvp_evk"])),
        },
        "session_paired": {
            "CV_P_delta_median": float(np.nanmedian(dcvp)),
            "CV_P_delta_mean": float(np.nanmean(dcvp)),
            "CV_P_sessions_spont_gt_evoked": int(np.sum(dcvp > 0)),
            "CV_P_wilcoxon_p_two_sided": float(wilcoxon(dcvp).pvalue),
            "S_P_delta_median": float(np.nanmedian(dsp)),
            "S_P_delta_mean": float(np.nanmean(dsp)),
            "S_P_sessions_spont_gt_evoked": int(np.sum(dsp > 0)),
            "S_P_wilcoxon_p_two_sided": float(wilcoxon(dsp).pvalue),
            "FR_delta_median_Hz": float(np.nanmedian(dfr)),
            "FR_sessions_spont_gt_evoked": int(np.sum(dfr > 0)),
            "FR_wilcoxon_p_two_sided": float(wilcoxon(dfr).pvalue),
        },
        "pooled_trial_medians": {
            "spontaneous_CV_P": float(np.nanmedian(pooled["cvp_spon"])),
            "evoked_CV_P": float(np.nanmedian(pooled["cvp_evk"])),
            "spontaneous_S_P": float(np.nanmedian(pooled["sp_spon"])),
            "evoked_S_P": float(np.nanmedian(pooled["sp_evk"])),
            "spontaneous_FR_Hz": float(np.nanmedian(pooled["fr_spon"])),
            "evoked_FR_Hz": float(np.nanmedian(pooled["fr_evk"])),
        },
        "pooled_trial_test": {
            "CV_P_spon_median_ci": list(boot_median_ci(pooled["cvp_spon"], seed=11)),
            "CV_P_evk_median_ci": list(boot_median_ci(pooled["cvp_evk"], seed=12)),
            "S_P_spon_median_ci": list(boot_median_ci(pooled["sp_spon"], seed=13)),
            "S_P_evk_median_ci": list(boot_median_ci(pooled["sp_evk"], seed=14)),
            "CV_P_mannwhitney_p_two_sided": float(
                mannwhitneyu(pooled["cvp_spon"], pooled["cvp_evk"], alternative="two-sided").pvalue
            ),
            "S_P_mannwhitney_p_two_sided": float(
                mannwhitneyu(pooled["sp_spon"], pooled["sp_evk"], alternative="two-sided").pvalue
            ),
            "note": (
                "Mann-Whitney computed on all 16,000 trials per state. Trials "
                "within a session are not independent, so these pooled p-values "
                "overstate significance (pseudoreplication); the honest paired "
                "test is the across-session Wilcoxon in 'session_paired'."
            ),
        },
        "interpretation": (
            "CV_P is a complementary population projection, not the primary "
            "Fig. 2 state axis. It is higher in spontaneous activity in 4/5 "
            "v1v2-1 sessions, but session 105l001p16 is the weak opposite "
            "case. CV_L remains the main V1 result."
        ),
    }
    return out


def downsample(x: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) <= max_n:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx], y[idx]


def _metric_inset(ax, pos, spon, evk, p, ylabel, top_pad=0.12):
    """Spontaneous-vs-evoked median bar inset (matches FigS1 D/E style).

    Bars are pooled medians; whiskers are the interquartile range (visible
    spread, unlike the near-zero bootstrap CI of the median); the bracket
    star is the pooled Mann-Whitney label (pseudoreplicated -- see caption).
    """
    spon = np.asarray(spon, dtype=float); spon = spon[np.isfinite(spon)]
    evk = np.asarray(evk, dtype=float); evk = evk[np.isfinite(evk)]
    med = [np.median(spon), np.median(evk)]
    q1 = [np.percentile(spon, 25), np.percentile(evk, 25)]
    q3 = [np.percentile(spon, 75), np.percentile(evk, 75)]
    yerr = np.vstack([np.array(med) - np.array(q1), np.array(q3) - np.array(med)])

    ins = ax.inset_axes(pos)
    ebar = dict(elinewidth=0.8, capsize=2.5, ecolor="#2b2b2b")
    ins.bar([0], [med[0]], 0.62, yerr=yerr[:, :1], color=BLUE,
            edgecolor="black", linewidth=0.5, error_kw=ebar)
    ins.bar([1], [med[1]], 0.62, yerr=yerr[:, 1:], color=ORANGE,
            edgecolor="black", linewidth=0.5, error_kw=ebar)

    # significance bracket
    ytop = max(q3)
    span = ytop - min(q1)
    ybr = ytop + 0.10 * span
    ins.plot([0, 0, 1, 1], [ybr, ybr + 0.04 * span, ybr + 0.04 * span, ybr],
             color="black", lw=0.8)
    ins.text(0.5, ybr + 0.04 * span, p_label(p), ha="center", va="bottom", fontsize=8)

    ylo = min(q1) - 0.10 * span
    yhi = ybr + 0.28 * span
    ins.set_ylim(ylo, yhi)
    ins.set_yticks([round(min(q1) + 0.05 * span, 2), round(max(q3), 2)])
    ins.set_xticks([])
    ins.set_xlim(-0.7, 1.7)
    ins.tick_params(labelsize=7, length=2)
    ins.set_ylabel(ylabel, fontsize=8, labelpad=1.0)
    for s in ("top", "right"):
        ins.spines[s].set_visible(False)
    return ins


def draw(rows: list[dict], pooled: dict[str, np.ndarray], summary: dict) -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    pt = summary["pooled_trial_test"]
    p_cvp = pt["CV_P_mannwhitney_p_two_sided"]
    p_sp = pt["S_P_mannwhitney_p_two_sided"]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))

    def panel_letter(ax, letter):
        ax.text(-0.20, 1.06, letter, transform=ax.transAxes, fontsize=20,
                family="serif", va="top", ha="left")

    # ---- Panel D: S_P vs firing rate ----
    ax = axes[0]
    xe, ye = downsample(pooled["fr_evk"], pooled["sp_evk"], 800, 2)
    xs, ys = downsample(pooled["fr_spon"], pooled["sp_spon"], 800, 1)
    ax.scatter(xe, ye, s=18, color=ORANGE, alpha=0.55, edgecolor="none", rasterized=True)
    ax.scatter(xs, ys, s=18, color=BLUE, alpha=0.55, edgecolor="none", rasterized=True)
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$S_P$")
    ax.set_xlim(0, 50)
    ax.set_ylim(0.2, 1.0)
    ax.set_xticks([0, 25, 50])
    ax.set_yticks([0.2, 1.0])
    panel_letter(ax, "D")
    _metric_inset(ax, [0.46, 0.58, 0.42, 0.40],
                  pooled["sp_spon"], pooled["sp_evk"], p_sp, r"$S_P$")

    # ---- Panel E: CV_P vs firing rate ----
    ax = axes[1]
    xe, ye = downsample(pooled["fr_evk"], pooled["cvp_evk"], 800, 4)
    xs, ys = downsample(pooled["fr_spon"], pooled["cvp_spon"], 800, 3)
    ax.scatter(xe, ye, s=18, color=ORANGE, alpha=0.55, edgecolor="none", rasterized=True)
    ax.scatter(xs, ys, s=18, color=BLUE, alpha=0.55, edgecolor="none", rasterized=True)
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$\mathrm{CV}_P$")
    ax.set_xlim(0, 50)
    ax.set_ylim(0.6, 2.0)
    ax.set_xticks([0, 25, 50])
    ax.set_yticks([0.6, 2.0])
    panel_letter(ax, "E")
    _metric_inset(ax, [0.46, 0.58, 0.42, 0.40],
                  pooled["cvp_spon"], pooled["cvp_evk"], p_cvp, r"$\mathrm{CV}_P$")

    # ---- Panel F: S_P-CV_P identity bands ----
    ax = axes[2]
    cv_grid = np.linspace(0, 3.0, 400)
    ax.plot(cv_grid, cv_grid**2 / (1 + cv_grid**2), color="black", lw=1.6, zorder=1)
    xs, ys = downsample(pooled["cvp_spon"], pooled["sp_spon"], 1000, 5)
    xe, ye = downsample(pooled["cvp_evk"], pooled["sp_evk"], 1000, 6)
    ax.scatter(xs, ys, s=22, color=BLUE, alpha=0.55, edgecolor="none", rasterized=True, zorder=2)
    ax.scatter(xe, ye, s=22, color=ORANGE, alpha=0.55, edgecolor="none", rasterized=True, zorder=3)
    ax.set_xlabel(r"$\mathrm{CV}_P$")
    ax.set_ylabel(r"$S_P$")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1.0])
    panel_letter(ax, "F")

    fig.tight_layout(pad=1.0, w_pad=2.2)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_LEGACY, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows, pooled = load_all_sessions()
    summary = summarize(rows, pooled)
    draw(rows, pooled, summary)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    if UPDATED_FIGURES.exists():
        shutil.copy2(OUT_PDF, UPDATED_FIGURES / OUT_PDF.name)
        shutil.copy2(OUT_PNG, UPDATED_FIGURES / OUT_PNG.name)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_LEGACY}")
    print(f"Saved {OUT_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
