"""Generate Fig. 2E (full-width, below C–D) — cross-dataset CV_L state contrast.

Matches the styling of Fig. 2 C–D panels:
  - Helvetica/Arial, base font 10 pt, xlabel 11 pt, ylabel 12 pt, ticks 9 pt.
  - Spines: left + bottom only, linewidth 0.9.
  - Full-width ~11″ landscape canvas.
  - Session/region summaries are displayed consistently as medians with 95% CIs.

Output
------
Fig2/Figure2E_cross_dataset_CVL.pdf
Fig2/Figure2E_cross_dataset_CVL.png
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.linewidth"] = 0.9

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
UPDATED_FIGURES = REPO / "updated" / "figures"

SUMMARY_JSON = BASE / "figure2E_cross_dataset_CVL_summary.json"
MACAQUE_NPZ = BASE / "Fig2_panels_CD_CVL_data.npz"
ALLEN_NPZ = BASE / "awake_V1_CVL.npz"
STEINMETZ_NPZ = BASE / "steinmetz_CVL_results.npz"

OUT_PDF = BASE / "Figure2E_cross_dataset_CVL.pdf"
OUT_PNG = BASE / "Figure2E_cross_dataset_CVL.png"

FR_MIN_HZ = 0.1

# ── colour palette (per data source) ────────────────────────────────
SOURCE_COLORS: dict[str, str] = {
    "Macaque V1":      "#4472C4",
    "Allen mouse":     "#70AD47",
    "Steinmetz mouse": "#5B9BD5",
    "IBL/BWM mouse":   "#9B59B6",
}

STEINMETZ_GROUPS = {
    "Visual cortex": ["VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa"],
    "Auditory":      ["AUD", "MG"],
    "Thalamus":      ["TH", "LGd", "LP", "MD", "PO", "POL", "LD", "VPL", "MG"],
    "Hippocampus":   ["CA1", "CA3", "DG", "SUB", "POST"],
    "Striatum/BG":   ["CP", "GPe"],
    "Frontal/PFC":   ["MOs", "ACA", "PL", "ILA", "ORBm"],
}

# ── short tick labels (disambiguated) ───────────────────────────────
TICK_LABELS: dict[str, str] = {
    "Macaque V1\n(CRCNS)":                   "V1\n(macaque)",
    "Mouse VISp\n(Allen)":                    "VISp\n(mouse AW)",
    "Mouse Visual cortex\n(Steinmetz)":       "Visual ctx\n(Steinmetz)",
    "Mouse Auditory\n(Steinmetz)":            "Auditory\n(Steinmetz)",
    "Mouse Thalamus\n(Steinmetz)":            "Thalamus\n(Steinmetz)",
    "Mouse Hippocampus\n(Steinmetz)":         "Hippocampus\n(Steinmetz)",
    "Mouse Striatum/BG\n(Steinmetz)":         "Striatum/BG\n(Steinmetz)",
    "Mouse Frontal/PFC\n(Steinmetz)":         "Frontal/PFC\n(Steinmetz)",
    "Mouse Thalamus\n(IBL/BWM)":              "Thalamus\n(IBL)",
    "Mouse Isocortex\n(IBL/BWM)":             "Isocortex\n(IBL)",
    "Mouse Midbrain\n(IBL/BWM)":              "Midbrain\n(IBL)",
    "Mouse Hindbrain\n(IBL/BWM)":             "Hindbrain\n(IBL)",
    "Mouse Cerebellum\n(IBL/BWM)":            "Cerebellum\n(IBL)",
    "Mouse HPF\n(IBL/BWM)":                   "HPF\n(IBL)",
}


def p_label(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "n.s."


# ══════════════════════════════════════════════════════════════════════
#  raw delta arrays for scatter
# ══════════════════════════════════════════════════════════════════════
def _load_raw_deltas() -> dict[str, np.ndarray]:
    deltas: dict[str, np.ndarray] = {}
    if MACAQUE_NPZ.exists():
        z = np.load(MACAQUE_NPZ)
        d = np.asarray(z["cvL_spon"], float) - np.asarray(z["cvL_evk"], float)
        deltas["Macaque V1 CRCNS"] = d[np.isfinite(d)]
    if ALLEN_NPZ.exists():
        z = np.load(ALLEN_NPZ)
        d = np.asarray(z["cvL_spon"], float) - np.asarray(z["cvL_evk"], float)
        deltas["Mouse VISp Allen"] = d[np.isfinite(d)]
    if STEINMETZ_NPZ.exists():
        z = np.load(STEINMETZ_NPZ, allow_pickle=True)
        for group, areas in STEINMETZ_GROUPS.items():
            chunks = []
            for area in areas:
                keys = [f"cvL_spon_{area}", f"cvL_evk_{area}",
                        f"FR_spon_{area}", f"FR_evk_{area}"]
                if not all(k in z.files for k in keys):
                    continue
                cs = np.asarray(z[f"cvL_spon_{area}"], float)
                ce = np.asarray(z[f"cvL_evk_{area}"], float)
                fs = np.asarray(z[f"FR_spon_{area}"], float)
                fe = np.asarray(z[f"FR_evk_{area}"], float)
                keep = (np.isfinite(cs) & np.isfinite(ce) & np.isfinite(fs)
                        & np.isfinite(fe) & (fs > FR_MIN_HZ) & (fe > FR_MIN_HZ))
                if keep.sum() >= 20:
                    chunks.append(cs[keep] - ce[keep])
            if chunks:
                deltas[f"Mouse {group} Steinmetz"] = np.concatenate(chunks)
    return deltas


# ══════════════════════════════════════════════════════════════════════
#  draw
# ══════════════════════════════════════════════════════════════════════
def draw(summaries: list[dict], raw_deltas: dict[str, np.ndarray]) -> None:
    n = len(summaries)
    fig_w = 11.0
    fig_h = max(3.2, 0.35 * n + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Keep the display scale tied to the reported session/region summaries.
    # Raw unit-level tails otherwise compress the cross-system effects near zero.
    ci_lows = np.asarray([s["delta_median_ci95"][0] for s in summaries], dtype=float)
    ci_highs = np.asarray([s["delta_median_ci95"][1] for s in summaries], dtype=float)
    y_lo = min(-0.55, float(np.nanmin(ci_lows)) - 0.08)
    y_hi = max(0.65, float(np.nanmax(ci_highs)) + 0.08)

    # ── zero line ──
    ax.axhline(0, color="0.4", lw=0.8, ls="--", zorder=0)

    x_positions = np.arange(n)
    xtick_labels = []

    for xi, s in zip(x_positions, summaries):
        color = SOURCE_COLORS.get(s["source"], "#888888")

        # tick label
        key = s["label"]
        xtick_labels.append(TICK_LABELS.get(key, key.replace("\n", " ")))

        # ── median diamond + CI ──
        med = s["delta_median"]
        lo, hi = s["delta_median_ci95"]
        ax.errorbar(
            xi, med,
            yerr=[[med - lo], [hi - med]],
            fmt="D",
            markersize=6.0,
            color="black",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            elinewidth=1.6,
            capsize=4.5,
            zorder=4,
        )

        # ── significance star above diamond ──
        p_val = s["inference_wilcoxon_p_two_sided"]
        star = p_label(p_val)
        star_y = hi + (y_hi - y_lo) * 0.06
        if star not in ("n.s.", "n/a"):
            ax.text(xi, star_y, star, ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
        else:
            ax.text(xi, star_y, star, ha="center", va="bottom",
                    fontsize=8, color="0.5")

    # ── axes (match C–D style) ──
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, fontsize=9, rotation=40, ha="right")
    ax.set_ylabel(r"$\Delta\,\mathrm{CV}_L$  (baseline $-$ driven)", fontsize=12)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.7, n - 0.3)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── legend ──
    handles = []
    for source, color in SOURCE_COLORS.items():
        if any(s["source"] == source for s in summaries):
            handles.append(plt.Line2D(
                [0], [0], marker="D", color="black",
                markerfacecolor=color, markersize=5.5,
                linestyle="None", label=source,
            ))
    ax.legend(handles=handles, frameon=False, fontsize=9,
              loc="lower right", bbox_to_anchor=(0.98, 0.02),
              ncol=2, borderaxespad=0.0)

    # ── footnote ──
    ax.text(
        0.5, -0.14,
        "Positive = higher baseline $\mathrm{CV}_L$.  "
        "Diamonds: pooled median ± 95% bootstrap CI.  "
        "Stars: paired Wilcoxon (Steinmetz & IBL clustered by session).",
        transform=ax.transAxes,
        ha="center", va="top", fontsize=7.6, color="0.35",
    )

    fig.tight_layout(pad=0.9)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(f"{SUMMARY_JSON} not found.")
    raw = json.loads(SUMMARY_JSON.read_text())
    raw_deltas = _load_raw_deltas()

    summaries: list[dict] = []
    for row in raw["rows"]:
        s = dict(row)
        label = s["label"]
        region = label.split("\n")[0] if "\n" in label else label
        tag = label.split("\n")[1] if "\n" in label else ""
        if tag in ("Steinmetz", "IBL/BWM", "CRCNS", "Allen"):
            s["label"] = f"{region}\n({tag})"
        else:
            s["label"] = region
        summaries.append(s)

    draw(summaries, raw_deltas)

    if UPDATED_FIGURES.exists():
        shutil.copy2(OUT_PDF, UPDATED_FIGURES / OUT_PDF.name)
        shutil.copy2(OUT_PNG, UPDATED_FIGURES / OUT_PNG.name)


if __name__ == "__main__":
    main()
