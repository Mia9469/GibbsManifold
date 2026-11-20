"""Generate a CV_P sanity-check companion for candidate Fig. 2E.

This is intentionally *not* the recommended main-text Fig. 2E panel.
CV_P samples across-neuron heterogeneity within a trial/window, whereas
CV_L samples across-trial variability of the same unit.  Because stimulus
or task engagement can increase across-neuron tuning heterogeneity, CV_P
can be weaker or even opposite in the single-session V1 view.

Outputs
-------
Fig2/Figure2E_CVP_sanity_check.pdf
Fig2/Figure2E_CVP_sanity_check.png
Fig2/figure2E_CVP_sanity_check_summary.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


BASE = Path(__file__).resolve().parent
REPO = BASE.parent
UPDATED_FIGURES = REPO / "updated" / "figures"

OUT_PDF = BASE / "Figure2E_CVP_sanity_check.pdf"
OUT_PNG = BASE / "Figure2E_CVP_sanity_check.png"
OUT_JSON = BASE / "figure2E_CVP_sanity_check_summary.json"

V1V2_FILES = [
    "105l001p16.mat",
    "106r001p26.mat",
    "106r002p70.mat",
    "107l002p67.mat",
    "107l003p143.mat",
]
IBL_OPS = REPO.parent / "IBL_data" / "ibl_metadata" / "gm_ops_pooled_v2.csv"

DUR_SPON = 1.5
DUR_DRIVEN = 1.28


def bootstrap_ci(values: np.ndarray, fn=np.nanmedian, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(0)
    boots = np.array([fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def load_rates(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    rasters = m["neuralData"][0, 0].spikeRasters
    spon = []
    evk = []
    for i in range(rasters.shape[0]):
        r = rasters[i, 0]
        if r.shape[1] == 1500:
            spon.append(np.asarray(r.sum(axis=1)).ravel() / DUR_SPON)
        elif r.shape[1] == 1280:
            evk.append(np.asarray(r.sum(axis=1)).ravel() / DUR_DRIVEN)
    return np.array(spon).T, np.array(evk).T


def cv_population(rate_matrix: np.ndarray) -> np.ndarray:
    mu = rate_matrix.mean(axis=0)
    return rate_matrix.std(axis=0) / np.where(mu > 1e-12, mu, np.nan)


def macaque_cvp_summary() -> dict:
    rows = []
    for filename in V1V2_FILES:
        spon, evk = load_rates(BASE / filename)
        cvp_s = cv_population(spon)
        cvp_e = cv_population(evk)
        diff = float(np.nanmedian(cvp_s) - np.nanmedian(cvp_e))
        rows.append({
            "session": filename.replace(".mat", ""),
            "n_units": int(spon.shape[0]),
            "n_spontaneous_trials": int(spon.shape[1]),
            "n_evoked_trials": int(evk.shape[1]),
            "spontaneous_median_CV_P": float(np.nanmedian(cvp_s)),
            "evoked_median_CV_P": float(np.nanmedian(cvp_e)),
            "delta_median_spont_minus_evoked": diff,
            "mannwhitney_p_two_sided": float(mannwhitneyu(cvp_s, cvp_e).pvalue),
        })
    deltas = np.array([r["delta_median_spont_minus_evoked"] for r in rows], dtype=float)
    return {
        "per_session": rows,
        "n_sessions": len(rows),
        "sessions_spont_gt_evoked": int(np.sum(deltas > 0)),
        "session_median_delta": float(np.nanmedian(deltas)),
        "session_mean_delta": float(np.nanmean(deltas)),
        "session_wilcoxon_p_two_sided": float(wilcoxon(deltas).pvalue),
    }


def ibl_cvp_summary() -> dict:
    ops = pd.read_csv(IBL_OPS)
    agg = (
        ops.groupby(["eid", "region", "condition"], as_index=False)["cv_P_median"]
        .median(numeric_only=True)
    )
    pv = agg.pivot(index=["eid", "region"], columns="condition", values="cv_P_median")
    pv = pv.dropna(subset=["spontaneous", "active_task"])
    delta = pv["spontaneous"].to_numpy(float) - pv["active_task"].to_numpy(float)
    med = float(np.nanmedian(delta))
    lo, hi = bootstrap_ci(delta, np.nanmedian)
    return {
        "n_pairs": int(len(delta)),
        "n_regions": int(len(set(idx[1] for idx in pv.index))),
        "spontaneous_median_CV_P": float(np.nanmedian(pv["spontaneous"].to_numpy(float))),
        "active_task_median_CV_P": float(np.nanmedian(pv["active_task"].to_numpy(float))),
        "delta_median_spont_minus_active": med,
        "delta_median_ci95": [lo, hi],
        "delta_mean_spont_minus_active": float(np.nanmean(delta)),
        "spontaneous_gt_active": int(np.sum(delta > 0)),
        "wilcoxon_p_two_sided": float(wilcoxon(delta).pvalue),
        "delta_values": delta.tolist(),
    }


def draw(macaque: dict, ibl: dict) -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45), gridspec_kw={"width_ratios": [1.05, 1.1]})
    ax = axes[0]
    rows = macaque["per_session"]
    deltas = np.array([r["delta_median_spont_minus_evoked"] for r in rows], dtype=float)
    x = np.arange(len(rows))
    colors = ["#C95F5F" if r["session"] == "105l001p16" else "#6E86B6" for r in rows]
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.75)
    ax.scatter(x, deltas, s=34, color=colors, edgecolor="black", linewidth=0.4, zorder=3)
    ax.plot(x, deltas, color="#999999", lw=0.8, alpha=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["session"].replace("l00", "l").replace("r00", "r") for r in rows], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(r"session median $\Delta \mathrm{CV}_P$")
    ax.set_title("A  Macaque V1 sessions", loc="left", fontweight="bold", fontsize=9)
    ax.text(
        0.02, 0.94,
        f"4/5 positive\npaired p={macaque['session_wilcoxon_p_two_sided']:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
    )

    ax = axes[1]
    delta = np.asarray(ibl["delta_values"], dtype=float)
    rng = np.random.default_rng(1)
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.75)
    ax.scatter(
        rng.normal(0, 0.055, len(delta)),
        delta,
        s=10,
        color="#9A6EA9",
        alpha=0.24,
        edgecolor="none",
        rasterized=True,
    )
    med = ibl["delta_median_spont_minus_active"]
    lo, hi = ibl["delta_median_ci95"]
    ax.errorbar(
        0,
        med,
        yerr=[[med - lo], [hi - med]],
        fmt="D",
        markersize=5.2,
        color="black",
        markerfacecolor="#9A6EA9",
        markeredgecolor="black",
        elinewidth=1.2,
        capsize=3.5,
        zorder=4,
    )
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([0])
    ax.set_xticklabels(["IBL/BWM\nregion pairs"], fontsize=8)
    ax.set_ylabel(r"$\Delta \mathrm{CV}_P$")
    ax.set_title("B  Awake mouse BWM", loc="left", fontweight="bold", fontsize=9)
    ax.text(
        0.03, 0.94,
        f"n={ibl['n_pairs']} pairs\nmedian={med:+.3f}\np={ibl['wilcoxon_p_two_sided']:.2g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
    )
    ax.set_ylim(-1.55, 1.55)
    axes[0].set_ylim(-0.12, 0.75)
    fig.suptitle(r"$\mathrm{CV}_P$ check: population axis is weaker/heterogeneous in V1", fontsize=9.2, y=1.02)
    fig.tight_layout(pad=0.7)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    macaque = macaque_cvp_summary()
    ibl = ibl_cvp_summary()
    out = {
        "interpretation": (
            "CV_P samples across-neuron heterogeneity within a trial/window. "
            "The single-session macaque V1 main dataset is slightly evoked-higher, "
            "whereas 4/5 macaque sessions and the IBL/BWM multi-region analysis "
            "show spontaneous/pre-task higher CV_P. This supports treating CV_L "
            "as the main Fig. 2E validation axis and keeping CV_P as a companion check."
        ),
        "allen_VISp_CV_P": (
            "Not computed from awake_V1_CVL.npz because that file stores per-unit "
            "CV_L/FR vectors only; CV_P would require recounting per-window population "
            "rates from the raw NWB."
        ),
        "macaque_V1": macaque,
        "IBL_BWM": {k: v for k, v in ibl.items() if k != "delta_values"},
    }
    draw(macaque, ibl)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    if UPDATED_FIGURES.exists():
        shutil.copy2(OUT_PDF, UPDATED_FIGURES / OUT_PDF.name)
        shutil.copy2(OUT_PNG, UPDATED_FIGURES / OUT_PNG.name)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_JSON}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
