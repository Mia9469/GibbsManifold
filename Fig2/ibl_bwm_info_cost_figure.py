"""Generate an IBL/BWM Fig. 5-style information-cost supplement.

This script extracts the state-specific information and cost quantities from
the IBL Brain-Wide Map operating-point table:

  - spontaneous: H(r) and C
  - active task: MI(s;r) and C

Active-task H(r) is shown only as a faint upper-bound/reference cloud, matching
the manuscript convention that evoked/task conditions should be interpreted
through measured MI rather than total response entropy.

Outputs
-------
Fig2/FigureS17_IBL_BWM_info_cost.pdf
Fig2/FigureS17_IBL_BWM_info_cost.png
Fig2/ibl_bwm_info_cost_summary.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon


BASE = Path(__file__).resolve().parent
REPO = BASE.parent
IBL_META = REPO.parent / "IBL_data" / "ibl_metadata"
OPS = IBL_META / "gm_ops_pooled_v2.csv"

OUT_PDF = BASE / "FigureS17_IBL_BWM_info_cost.pdf"
OUT_PNG = BASE / "FigureS17_IBL_BWM_info_cost.png"
OUT_JSON = BASE / "ibl_bwm_info_cost_summary.json"


COL_SPON = "#4C88C8"
COL_TASK = "#D07A45"
COL_TASK_H = "#888888"


def clean_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def state_summary(df: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for condition, label, info_col, ice_col in [
        ("spontaneous", "spontaneous_H", "Hr_median_bits_s", "ICE_H_median_bits_per_J"),
        ("active_task", "active_MI", "MI_median_bits_s", "ICE_I_median_bits_per_J"),
        ("active_task", "active_H_upper_bound", "Hr_median_bits_s", "ICE_H_median_bits_per_J"),
    ]:
        sub = df[df["condition"].eq(condition)].dropna(subset=[info_col, "C_median_W"])
        out[label] = {
            "n_rows": int(len(sub)),
            "n_regions": int(sub["region"].nunique()),
            "n_sessions": int(sub["eid"].nunique()),
            "info_col": info_col,
            "cost_col": "C_median_W",
            "ice_col": ice_col,
            "info_median_bits_s": float(np.nanmedian(sub[info_col])),
            "info_iqr_bits_s": [
                float(np.nanpercentile(sub[info_col], 25)),
                float(np.nanpercentile(sub[info_col], 75)),
            ],
            "cost_median_W": float(np.nanmedian(sub["C_median_W"])),
            "cost_iqr_W": [
                float(np.nanpercentile(sub["C_median_W"], 25)),
                float(np.nanpercentile(sub["C_median_W"], 75)),
            ],
            "ice_median_bits_per_J": float(np.nanmedian(sub[ice_col])),
            "ice_iqr_bits_per_J": [
                float(np.nanpercentile(sub[ice_col], 25)),
                float(np.nanpercentile(sub[ice_col], 75)),
            ],
        }
    return out


def paired_summary(df: pd.DataFrame) -> dict:
    metrics = [
        "C_median_W",
        "Hr_median_bits_s",
        "MI_median_bits_s",
        "ICE_H_median_bits_per_J",
        "ICE_I_median_bits_per_J",
    ]
    agg = (
        df.groupby(["eid", "region", "condition"], as_index=False)[metrics]
        .median(numeric_only=True)
    )
    pv = agg.pivot(index=["eid", "region"], columns="condition", values=metrics)

    rows = []
    for idx in pv.index:
        required = [
            ("C_median_W", "spontaneous"),
            ("Hr_median_bits_s", "spontaneous"),
            ("ICE_H_median_bits_per_J", "spontaneous"),
            ("C_median_W", "active_task"),
            ("Hr_median_bits_s", "active_task"),
            ("MI_median_bits_s", "active_task"),
            ("ICE_H_median_bits_per_J", "active_task"),
            ("ICE_I_median_bits_per_J", "active_task"),
        ]
        if not all(key in pv.columns and np.isfinite(pv.loc[idx, key]) for key in required):
            continue
        rows.append({
            "eid": idx[0],
            "region": idx[1],
            "C_spon": pv.loc[idx, ("C_median_W", "spontaneous")],
            "H_spon": pv.loc[idx, ("Hr_median_bits_s", "spontaneous")],
            "HC_spon": pv.loc[idx, ("ICE_H_median_bits_per_J", "spontaneous")],
            "C_task": pv.loc[idx, ("C_median_W", "active_task")],
            "H_task": pv.loc[idx, ("Hr_median_bits_s", "active_task")],
            "MI_task": pv.loc[idx, ("MI_median_bits_s", "active_task")],
            "HC_task": pv.loc[idx, ("ICE_H_median_bits_per_J", "active_task")],
            "MIC_task": pv.loc[idx, ("ICE_I_median_bits_per_J", "active_task")],
        })
    paired = pd.DataFrame(rows)

    out = {"n_pairs": int(len(paired)), "n_regions": int(paired["region"].nunique())}
    for col in ["C_spon", "H_spon", "HC_spon", "C_task", "H_task", "MI_task", "HC_task", "MIC_task"]:
        vals = paired[col].to_numpy(float)
        out[col] = {
            "median": float(np.nanmedian(vals)),
            "iqr": [
                float(np.nanpercentile(vals, 25)),
                float(np.nanpercentile(vals, 75)),
            ],
        }
    for active_col, spon_col, name in [
        ("C_task", "C_spon", "delta_C_task_minus_spon"),
        ("H_task", "H_spon", "delta_H_task_minus_spon"),
        ("HC_task", "HC_spon", "delta_HC_task_minus_spon"),
        ("MIC_task", "HC_spon", "delta_MIC_task_minus_HC_spon"),
    ]:
        delta = paired[active_col].to_numpy(float) - paired[spon_col].to_numpy(float)
        n_pos = int(np.sum(delta > 0))
        out[name] = {
            "median": float(np.nanmedian(delta)),
            "mean": float(np.nanmean(delta)),
            "positive": n_pos,
            "fraction_positive": float(n_pos / len(delta)),
            "wilcoxon_p_two_sided": float(wilcoxon(delta).pvalue),
            "sign_p_two_sided": float(binomtest(n_pos, len(delta), 0.5).pvalue),
        }

    ratio_specs = [
        ("C_task", "C_spon", "ratio_C_task_over_spon"),
        ("MI_task", "H_spon", "ratio_MI_task_over_H_spon"),
        ("MIC_task", "HC_spon", "ratio_MIC_task_over_HC_spon"),
    ]
    for num_col, den_col, name in ratio_specs:
        num = paired[num_col].to_numpy(float)
        den = paired[den_col].to_numpy(float)
        ratio = num / den
        out[name] = {
            "median": float(np.nanmedian(ratio)),
            "iqr": [
                float(np.nanpercentile(ratio, 25)),
                float(np.nanpercentile(ratio, 75)),
            ],
            "gt_one": int(np.sum(ratio > 1.0)),
            "fraction_gt_one": float(np.mean(ratio > 1.0)),
        }
    return out


def draw(df: pd.DataFrame, summary: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    spon = df[df["condition"].eq("spontaneous")].copy()
    task = df[df["condition"].eq("active_task")].copy()

    fig = plt.figure(figsize=(13.5, 4.7))
    gs = fig.add_gridspec(1, 3, left=0.065, right=0.985, bottom=0.18, top=0.84, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel A: information vs cost.
    ax_a.scatter(
        task["C_median_W"], task["Hr_median_bits_s"],
        s=18, color=COL_TASK_H, alpha=0.22, label="Active H(r) upper bound"
    )
    ax_a.scatter(
        spon["C_median_W"], spon["Hr_median_bits_s"],
        s=24, facecolors="none", edgecolors=COL_SPON, linewidths=0.8,
        alpha=0.85, label="Spontaneous H(r)"
    )
    ax_a.scatter(
        task["C_median_W"], task["MI_median_bits_s"],
        s=24, color=COL_TASK, alpha=0.82, label="Active task MI(s;r)"
    )
    ax_a.set_xlabel("calibrated cost proxy")
    ax_a.set_ylabel("Information rate [bits/s]")
    ax_a.set_title("A  BWM information-cost operating points", fontsize=10)
    ax_a.legend(fontsize=7.5, frameon=False, loc="upper left")

    # Panel B: ICE vs cost.
    ax_b.scatter(
        task["C_median_W"], task["ICE_H_median_bits_per_J"],
        s=18, color=COL_TASK_H, alpha=0.22, label="Active H/C upper bound"
    )
    ax_b.scatter(
        spon["C_median_W"], spon["ICE_H_median_bits_per_J"],
        s=24, facecolors="none", edgecolors=COL_SPON, linewidths=0.8,
        alpha=0.85, label="Spontaneous H/C"
    )
    ax_b.scatter(
        task["C_median_W"], task["ICE_I_median_bits_per_J"],
        s=24, color=COL_TASK, alpha=0.82, label="Active MI/C"
    )
    ax_b.set_xlabel("calibrated cost proxy")
    ax_b.set_ylabel("information / calibrated cost proxy")
    ax_b.set_title("B  Cost-normalized operating points", fontsize=10)
    ax_b.legend(fontsize=7.5, frameon=False, loc="upper right")

    # Panel C: paired dimensionless ratios for the state-specific quantities.
    paired = summary["paired"]
    labels = [
        "C$_{task}$ /\nC$_{spon}$",
        "MI$_{task}$ /\nH$_{spon}$",
        "(MI/C)$_{task}$ /\n(H/C)$_{spon}$",
    ]
    ratio_keys = [
        "ratio_C_task_over_spon",
        "ratio_MI_task_over_H_spon",
        "ratio_MIC_task_over_HC_spon",
    ]
    vals = [paired[k]["median"] for k in ratio_keys]
    iqrs = [paired[k]["iqr"] for k in ratio_keys]
    x = np.arange(len(labels))
    lower = [v - q[0] for v, q in zip(vals, iqrs)]
    upper = [q[1] - v for v, q in zip(vals, iqrs)]
    ax_c.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.65)
    ax_c.bar(x, vals, 0.55, color=[COL_TASK, "#A86DB7", "#6B8F3A"], alpha=0.85)
    ax_c.errorbar(x, vals, yerr=[lower, upper], fmt="none",
                  ecolor="k", elinewidth=1.1, capsize=3)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, fontsize=8)
    ax_c.set_title("C  Paired ratios (163 region-session pairs)", fontsize=10)
    ax_c.set_ylabel("Median ratio (IQR)")
    ax_c.set_ylim(0, max(1.55, max(v + u for v, u in zip(vals, upper)) * 1.12))
    ax_c.text(
        0.04, 0.95,
        "Numerators are state-specific:\n"
        "spontaneous = H(r)\n"
        "active task = MI(s;r)\n"
        "dashed line = ratio 1",
        transform=ax_c.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9)
    )

    fig.suptitle(
        "IBL Brain-Wide Map: Fig. 5-style H/C and MI/C operating-point quantities",
        fontsize=11, y=0.965
    )

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    if not OPS.exists():
        raise FileNotFoundError(OPS)
    df = pd.read_csv(OPS)
    summary = {
        "source_table": str(OPS),
        "state_summary": state_summary(df),
        "paired": paired_summary(df),
        "definition_note": (
            "Spontaneous rows use Hr_median_bits_s and C_median_W. "
            "Active-task rows use MI_median_bits_s and C_median_W; "
            "active Hr is plotted only as an entropy upper-bound reference."
        ),
    }
    draw(df, summary)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    updated_figures = REPO / "updated" / "figures"
    if updated_figures.exists():
        shutil.copy2(OUT_PDF, updated_figures / OUT_PDF.name)
        shutil.copy2(OUT_PNG, updated_figures / OUT_PNG.name)

    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_JSON}")
    print(json.dumps(summary["state_summary"], indent=2))
    print(json.dumps(summary["paired"], indent=2))


if __name__ == "__main__":
    main()
