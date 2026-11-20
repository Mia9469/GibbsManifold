"""Generate an IBL/BWM spontaneous-to-task state-transition supplement.

This figure reuses the Fig. 2 CV_L / S_L analysis logic on the IBL
Brain-Wide Map operating-point table.  The table contains matched
spontaneous and active-task measurements for the same (eid, region)
entries, with 0.4 s windows in both states.

Outputs
-------
Fig2/FigureS16_IBL_BWM_state_transition.pdf
Fig2/FigureS16_IBL_BWM_state_transition.png
Fig2/ibl_bwm_state_transition_summary.json

The script also copies the figure files into updated/figures/ when that
directory exists.
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
REGION_METRICS = IBL_META / "learning_state_transition_region_metrics.csv"

OUT_PDF = BASE / "FigureS16_IBL_BWM_state_transition.pdf"
OUT_PNG = BASE / "FigureS16_IBL_BWM_state_transition.png"
OUT_JSON = BASE / "ibl_bwm_state_transition_summary.json"


METRICS = [
    "cv_L_median",
    "s_L_median",
    "cv_P_median",
    "s_P_median",
    "FR_median_hz",
    "Hr_median_bits_s",
    "ICE_H_median_bits_per_J",
    "PR_dim",
]

GROUP_COLORS = {
    "Isocortex": "#C75D6A",
    "Thalamus": "#4C88C8",
    "HPF": "#7A63B8",
    "Midbrain": "#6AAE75",
    "Hindbrain": "#9C7A3A",
    "Cerebellum": "#D07A45",
    "Other": "#777777",
}

ISOCORTEX_PREFIXES = ("VIS", "SS", "MO", "ACA", "RSP", "AUD", "FRP", "PL", "ILA",
                      "ORB", "TEa", "ECT", "PERI", "AI", "GU", "VISC", "PTLp")
HPF_PREFIXES = ("CA", "DG", "SUB", "POST", "PRE", "APr", "ENT", "HATA", "HPF")
THALAMUS_PREFIXES = ("LGd", "LGv", "LP", "LD", "LH", "VPL", "VPM", "MD", "MG", "PO",
                     "RT", "AV", "AM", "AD", "VAL", "VM", "CL", "PCN", "PF", "TH",
                     "IGL", "IntG", "RE", "Xi", "SMT")
MIDBRAIN_PREFIXES = ("SCm", "SCs", "SCdg", "SCig", "SCop", "SCzo", "MRN", "RN",
                     "APN", "PAG", "MB", "IC", "VTA", "SNr", "SNc", "PPN",
                     "IPN", "IPC", "IPDL", "IPI", "IPL", "PRT", "OP", "InG", "InC")
HINDBRAIN_PREFIXES = ("P", "PG", "PRN", "PB", "P5", "SUT", "SOC", "GRN", "MV",
                      "LRN", "HB", "MARN", "NTS", "ICP", "SPVO", "DCO", "Vest",
                      "CN", "CU")
CEREBELLUM_PREFIXES = ("CBX", "CB", "CENT", "ANcr", "Cul", "CUL", "PRM", "SIM",
                       "FN", "IP", "DN", "PYR", "arb", "FOTU")


def cv_to_s(cv: np.ndarray) -> np.ndarray:
    return cv**2 / (1.0 + cv**2)


def assign_group(region: str) -> str:
    if not isinstance(region, str):
        return "Other"
    # PAR is parasubiculum; PARN is a hindbrain reticular nucleus.
    if region == "PAR":
        return "HPF"
    for prefixes, name in [
        (ISOCORTEX_PREFIXES, "Isocortex"),
        (HPF_PREFIXES, "HPF"),
        (THALAMUS_PREFIXES, "Thalamus"),
        (CEREBELLUM_PREFIXES, "Cerebellum"),
        (MIDBRAIN_PREFIXES, "Midbrain"),
        (HINDBRAIN_PREFIXES, "Hindbrain"),
    ]:
        if region.startswith(prefixes):
            return name
    return "Other"


def safe_wilcoxon(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0 or not np.any(delta != 0):
        return float("nan")
    return float(wilcoxon(delta).pvalue)


def paired_table(ops: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRICS if m in ops.columns]
    agg = (
        ops.groupby(["eid", "region", "condition"], as_index=False)[metrics]
        .median(numeric_only=True)
    )
    pv = agg.pivot(index=["eid", "region"], columns="condition", values=metrics)

    rows = []
    for idx in pv.index:
        row = {"eid": idx[0], "region": idx[1]}
        ok = True
        for metric in metrics:
            active_key = (metric, "active_task")
            spont_key = (metric, "spontaneous")
            if active_key not in pv.columns or spont_key not in pv.columns:
                ok = False
                break
            row[f"{metric}_active"] = pv.loc[idx, active_key]
            row[f"{metric}_spont"] = pv.loc[idx, spont_key]
            row[f"{metric}_delta_spont_minus_active"] = (
                pv.loc[idx, spont_key] - pv.loc[idx, active_key]
            )
        if ok:
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.dropna(subset=["cv_L_median_active", "cv_L_median_spont"])


def metric_summary(df: pd.DataFrame, metric: str) -> dict:
    active = df[f"{metric}_active"].to_numpy(float)
    spont = df[f"{metric}_spont"].to_numpy(float)
    mask = np.isfinite(active) & np.isfinite(spont)
    active = active[mask]
    spont = spont[mask]
    delta = spont - active
    n_pos = int(np.sum(delta > 0))
    tmp = df.loc[mask, ["eid"]].copy()
    tmp["delta"] = delta
    session_delta = tmp.groupby("eid")["delta"].median().to_numpy(float)
    session_pos = int(np.sum(session_delta > 0))
    return {
        "n_pairs": int(delta.size),
        "n_regions": int(df.loc[mask, "region"].nunique()),
        "active_median": float(np.nanmedian(active)),
        "spontaneous_median": float(np.nanmedian(spont)),
        "delta_spont_minus_active_median": float(np.nanmedian(delta)),
        "delta_spont_minus_active_mean": float(np.nanmean(delta)),
        "spont_gt_active": n_pos,
        "spont_gt_active_fraction": float(n_pos / delta.size),
        "observation_level_wilcoxon_p_two_sided_descriptive": safe_wilcoxon(delta),
        "observation_level_sign_p_two_sided_descriptive": float(binomtest(n_pos, delta.size, 0.5).pvalue),
        "n_sessions": int(len(session_delta)),
        "session_median_delta": float(np.nanmedian(session_delta)),
        "session_cluster_wilcoxon_p_two_sided": safe_wilcoxon(session_delta),
        "session_cluster_sign_p_two_sided": float(
            binomtest(session_pos, len(session_delta), 0.5).pvalue
        ),
    }


def load_region_cv() -> pd.DataFrame:
    if not REGION_METRICS.exists():
        raise FileNotFoundError(REGION_METRICS)
    reg = pd.read_csv(REGION_METRICS)
    cv = reg[reg["metric"].eq("cv_L_median")].copy()
    cv["delta_spont_minus_active"] = -cv["delta_active_minus_spont_median"]
    return cv.sort_values("delta_spont_minus_active", ascending=False)


def make_figure(paired: pd.DataFrame, region_cv: pd.DataFrame, summary: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(14.5, 4.8))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 1.15, 1.2],
        left=0.065, right=0.985, bottom=0.20, top=0.84, wspace=0.38
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel A: paired region-session CV_L.
    active = paired["cv_L_median_active"].to_numpy(float)
    spont = paired["cv_L_median_spont"].to_numpy(float)
    groups = paired["group"].fillna("Other").to_numpy(str)
    for group in sorted(set(groups)):
        mask = groups == group
        ax_a.scatter(
            active[mask], spont[mask], s=25,
            color=GROUP_COLORS.get(group, "#777777"),
            alpha=0.75, edgecolor="white", linewidth=0.35, label=group
        )
    lim = [0, max(np.nanmax(active), np.nanmax(spont), 2.2) * 1.04]
    ax_a.plot(lim, lim, "k--", lw=1.0, alpha=0.7)
    ax_a.set_xlim(lim)
    ax_a.set_ylim(lim)
    ax_a.set_xlabel("Active-task CV$_L$")
    ax_a.set_ylabel("Spontaneous CV$_L$")
    cv_sum = summary["cv_L_median"]
    ax_a.set_title("A  Matched IBL/BWM state pairs", fontsize=10)
    ax_a.text(
        0.04, 0.96,
        f"n={cv_sum['n_pairs']} pairs, {cv_sum['n_regions']} regions\n"
        f"median delta={cv_sum['delta_spont_minus_active_median']:+.3f}\n"
        f"session-cluster p={cv_sum['session_cluster_wilcoxon_p_two_sided']:.2g}",
        transform=ax_a.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9)
    )

    handles, labels = ax_a.get_legend_handles_labels()
    order = [i for i, label in enumerate(labels) if label in GROUP_COLORS]
    if order:
        ax_a.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            fontsize=7, frameon=False, loc="lower right", ncol=1
        )

    # Panel B: region-level deltas for regions with at least 3 matched sessions.
    y = np.arange(len(region_cv))
    colors = [GROUP_COLORS.get(g, "#777777") for g in region_cv["group"]]
    ax_b.axvline(0, color="k", lw=1.0, alpha=0.7)
    ax_b.barh(y, region_cv["delta_spont_minus_active"], color=colors, alpha=0.85)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(region_cv["region"], fontsize=8)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Delta CV$_L$ (spontaneous - active)")
    ax_b.set_title("B  Brain-region medians (>=3 sessions)", fontsize=10)
    xlim = max(abs(region_cv["delta_spont_minus_active"]).max() * 1.20, 0.55)
    ax_b.set_xlim(-xlim, xlim)
    for yi, (_, row) in enumerate(region_cv.iterrows()):
        ax_b.text(
            row["delta_spont_minus_active"] + (0.015 if row["delta_spont_minus_active"] >= 0 else -0.015),
            yi, f"n={int(row['n_pairs'])}",
            va="center",
            ha="left" if row["delta_spont_minus_active"] >= 0 else "right",
            fontsize=7,
        )

    # Panel C: S_L-CV_L operating points and Fig. 2 identity.
    cv_grid = np.linspace(0, 2.4, 300)
    ax_c.plot(cv_grid, cv_to_s(cv_grid), "k-", lw=1.6, label="S = CV$^2$/(1+CV$^2$)")
    for state, marker, fc, ec, alpha in [
        ("spont", "o", "none", "#4C88C8", 0.72),
        ("active", "o", "#D07A45", "#D07A45", 0.58),
    ]:
        cv = paired[f"cv_L_median_{state}"].to_numpy(float)
        sl = paired[f"s_L_median_{state}"].to_numpy(float)
        mask = np.isfinite(cv) & np.isfinite(sl)
        ax_c.scatter(
            cv[mask], sl[mask], s=18, marker=marker,
            facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=0.65,
            label="Spontaneous" if state == "spont" else "Active task"
        )
    ax_c.set_xlabel("CV$_L$")
    ax_c.set_ylabel("$S_L$")
    ax_c.set_xlim(0, 2.4)
    ax_c.set_ylim(0, 0.9)
    ax_c.set_title("C  Same CV-sparseness identity as Fig. 2", fontsize=10)
    ax_c.legend(fontsize=8, frameon=False, loc="lower right")

    fig.suptitle(
        "IBL Brain-Wide Map: state-dependent CV and sparseness are regionally heterogeneous",
        fontsize=11, y=0.965
    )

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    if not OPS.exists():
        raise FileNotFoundError(OPS)

    ops = pd.read_csv(OPS)
    paired = paired_table(ops)

    paired["group"] = paired["region"].map(assign_group)

    region_cv = load_region_cv()
    summary = {metric: metric_summary(paired, metric) for metric in METRICS if f"{metric}_active" in paired}
    summary["region_cv_L_median_ge3_sessions"] = region_cv.to_dict(orient="records")

    make_figure(paired, region_cv, summary)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    updated_figures = REPO / "updated" / "figures"
    if updated_figures.exists():
        shutil.copy2(OUT_PDF, updated_figures / OUT_PDF.name)
        shutil.copy2(OUT_PNG, updated_figures / OUT_PNG.name)

    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_JSON}")
    print(json.dumps({
        "cv_L": summary["cv_L_median"],
        "s_L": summary["s_L_median"],
        "cv_P": summary["cv_P_median"],
        "s_P": summary["s_P_median"],
    }, indent=2))


if __name__ == "__main__":
    main()
