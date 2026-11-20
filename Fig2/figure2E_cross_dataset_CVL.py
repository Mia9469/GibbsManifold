"""Generate candidate main-text Fig. 2E multi-region CV_L state contrast.

The panel shows the matched state difference

    Delta CV_L = spontaneous/pre-task CV_L - evoked/active-task CV_L

for datasets with comparable state definitions:
  1. macaque V1 under sufentanil anesthesia (main Fig. 2 dataset),
  2. awake mouse VISp from Allen Visual Coding Neuropixels,
  3. awake mouse Steinmetz/NMA areas grouped into broad brain systems,
  4. awake trained mouse IBL Brain-Wide Map grouped into broad brain systems.

Cat pvc-8 is not included because it has evoked data only.  Macaque pvc-11
is not included because spontaneous and evoked recordings are unpaired
sessions and therefore do not support the same matched state contrast.

Outputs
-------
Fig2/Figure2E_cross_dataset_CVL.pdf
Fig2/Figure2E_cross_dataset_CVL.png
Fig2/figure2E_cross_dataset_CVL_summary.json

The PDF/PNG are also copied into updated/figures/ when that directory exists.
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
UPDATED_FIGURES = REPO / "updated" / "figures"

OUT_PDF = BASE / "Figure2E_cross_dataset_CVL.pdf"
OUT_PNG = BASE / "Figure2E_cross_dataset_CVL.png"
OUT_JSON = BASE / "figure2E_cross_dataset_CVL_summary.json"

MACAQUE_NPZ = BASE / "Fig2_panels_CD_CVL_data.npz"
ALLEN_NPZ = BASE / "awake_V1_CVL.npz"
STEINMETZ_NPZ = BASE / "steinmetz_CVL_results.npz"
IBL_OPS = REPO.parent / "IBL_data" / "ibl_metadata" / "gm_ops_pooled_v2.csv"

FR_MIN_HZ = 0.1

SOURCE_COLORS = {
    "Macaque V1": "#5B6F95",
    "Allen mouse": "#5A9E6F",
    "Steinmetz mouse": "#4C88C8",
    "IBL/BWM mouse": "#9A6EA9",
}

STEINMETZ_GROUPS = {
    "Visual cortex": ["VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa"],
    "Auditory": ["AUD", "MG"],
    "Thalamus": ["TH", "LGd", "LP", "MD", "PO", "POL", "LD", "VPL", "MG"],
    "Hippocampus": ["CA1", "CA3", "DG", "SUB", "POST"],
    "Striatum/BG": ["CP", "GPe"],
    "Frontal/PFC": ["MOs", "ACA", "PL", "ILA", "ORBm"],
}

ISOCORTEX_PREFIXES = (
    "VIS", "SS", "MO", "ACA", "RSP", "AUD", "FRP", "PL", "ILA",
    "ORB", "TEa", "ECT", "PERI", "AI", "GU", "VISC", "PTLp",
)
HPF_PREFIXES = ("CA", "DG", "SUB", "POST", "PRE", "APr", "ENT", "HATA", "HPF")
THALAMUS_PREFIXES = (
    "LGd", "LGv", "LP", "LD", "LH", "VPL", "VPM", "MD", "MG", "PO",
    "RT", "AV", "AM", "AD", "VAL", "VM", "CL", "PCN", "PF", "TH",
    "IGL", "IntG", "RE", "Xi", "SMT",
)
MIDBRAIN_PREFIXES = (
    "SCm", "SCs", "SCdg", "SCig", "SCop", "SCzo", "MRN", "RN",
    "APN", "PAG", "MB", "IC", "VTA", "SNr", "SNc", "PPN",
    "IPN", "IPC", "IPDL", "IPI", "IPL", "PRT", "OP", "InG", "InC",
)
HINDBRAIN_PREFIXES = (
    "P", "PG", "PRN", "PB", "P5", "SUT", "SOC", "GRN", "MV",
    "LRN", "HB", "MARN", "NTS", "ICP", "SPVO", "DCO", "Vest",
    "CN", "CU",
)
CEREBELLUM_PREFIXES = (
    "CBX", "CB", "CENT", "ANcr", "Cul", "CUL", "PRM", "SIM",
    "FN", "IP", "DN", "PYR", "arb", "FOTU",
)

IBL_GROUP_ORDER = ["Thalamus", "Isocortex", "Midbrain", "Hindbrain", "Cerebellum", "HPF"]


def assign_ibl_group(region: str) -> str:
    if not isinstance(region, str):
        return "Other"
    # PAR is parasubiculum, whereas PARN is the parvicellular reticular
    # nucleus in the hindbrain; a generic PAR prefix would conflate them.
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


def bootstrap_ci(values: np.ndarray, fn=np.nanmedian, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boots = np.array([fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 10000) -> tuple[float, float]:
    """Bootstrap the pooled median by resampling independent clusters."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    keep = np.isfinite(values)
    values, clusters = values[keep], clusters[keep]
    unique = np.unique(clusters)
    if not len(unique):
        return float("nan"), float("nan")
    by_cluster = {c: values[clusters == c] for c in unique}
    rng = np.random.default_rng(0)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        selected = rng.choice(unique, size=len(unique), replace=True)
        boots[b] = np.nanmedian(np.concatenate([by_cluster[c] for c in selected]))
    return tuple(map(float, np.nanpercentile(boots, [2.5, 97.5])))


def p_label(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def summarize(
    label: str,
    source: str,
    spontaneous: np.ndarray,
    evoked: np.ndarray,
    unit: str,
    notes: str = "",
    clusters: np.ndarray | None = None,
) -> dict:
    spontaneous = np.asarray(spontaneous, dtype=float)
    evoked = np.asarray(evoked, dtype=float)
    keep = np.isfinite(spontaneous) & np.isfinite(evoked)
    spontaneous = spontaneous[keep]
    evoked = evoked[keep]
    delta = spontaneous - evoked
    if clusters is not None:
        clusters = np.asarray(clusters)[keep]
        unique_clusters = np.unique(clusters)
        cluster_delta = np.array([np.nanmedian(delta[clusters == c]) for c in unique_clusters])
        inference_values = cluster_delta
        med_ci = cluster_bootstrap_ci(delta, clusters)
        inference_level = "cluster"
    else:
        unique_clusters = np.array([])
        inference_values = delta
        med_ci = bootstrap_ci(delta, np.nanmedian)
        inference_level = "observation"
    n_pos = int(np.sum(delta > 0))
    if inference_values.size == 0:
        w_p = float("nan")
        sign_p = float("nan")
    elif np.any(inference_values != 0):
        w_p = float(wilcoxon(inference_values).pvalue)
        n_cluster_pos = int(np.sum(inference_values > 0))
        sign_p = float(binomtest(n_cluster_pos, len(inference_values), 0.5).pvalue)
    else:
        w_p = 1.0
        sign_p = 1.0
    mean_ci = bootstrap_ci(delta, np.nanmean)
    return {
        "label": label,
        "source": source,
        "unit": unit,
        "n": int(len(delta)),
        "inference_level": inference_level,
        "n_clusters": int(len(unique_clusters)) if clusters is not None else None,
        "spontaneous_median": float(np.nanmedian(spontaneous)) if len(spontaneous) else float("nan"),
        "evoked_or_task_median": float(np.nanmedian(evoked)) if len(evoked) else float("nan"),
        "delta": delta.tolist(),
        "delta_median": float(np.nanmedian(delta)) if len(delta) else float("nan"),
        "delta_median_ci95": list(med_ci),
        "delta_mean": float(np.nanmean(delta)) if len(delta) else float("nan"),
        "delta_mean_ci95": list(mean_ci),
        "spontaneous_gt_evoked_or_task": n_pos,
        "spontaneous_gt_fraction": float(n_pos / len(delta)) if len(delta) else float("nan"),
        "inference_wilcoxon_p_two_sided": w_p,
        "inference_sign_p_two_sided": sign_p,
        "notes": notes,
    }


def load_macaque() -> dict:
    if not MACAQUE_NPZ.exists():
        raise FileNotFoundError(MACAQUE_NPZ)
    z = np.load(MACAQUE_NPZ)
    return summarize(
        "Macaque V1\nCRCNS",
        "Macaque V1",
        z["cvL_spon"],
        z["cvL_evk"],
        "neurons",
        "paired neurons from the main Fig. 2 macaque V1 dataset",
    )


def load_allen() -> dict:
    if not ALLEN_NPZ.exists():
        raise FileNotFoundError(ALLEN_NPZ)
    z = np.load(ALLEN_NPZ)
    if "spon_window_s" not in z.files or "evk_window_s" not in z.files:
        raise RuntimeError(
            "awake_V1_CVL.npz predates matched-window metadata. Re-run "
            "awake_V1_CVL_confirmation.py with the Allen NWB before regenerating Fig. 2E."
        )
    spon_window_s = float(z["spon_window_s"])
    evk_window_s = float(z["evk_window_s"])
    if not np.isclose(spon_window_s, evk_window_s):
        raise RuntimeError(
            "Allen CV_L windows are unequal. Re-run awake_V1_CVL_confirmation.py "
            "before regenerating Fig. 2E."
        )
    notes = (
        "awake mouse VISp; paired units from Allen Visual Coding Neuropixels; "
        f"matched {spon_window_s:g} s windows"
    )
    return summarize(
        "Mouse VISp\nAllen",
        "Allen mouse",
        z["cvL_spon"],
        z["cvL_evk"],
        "units",
        notes,
    )


def load_steinmetz_groups() -> list[dict]:
    if not STEINMETZ_NPZ.exists():
        raise FileNotFoundError(STEINMETZ_NPZ)
    z = np.load(STEINMETZ_NPZ, allow_pickle=True)
    out = []
    for group, areas in STEINMETZ_GROUPS.items():
        deltas_s = []
        deltas_e = []
        session_ids = []
        used_areas = []
        for area in areas:
            keys = [
                f"cvL_spon_{area}", f"cvL_evk_{area}",
                f"FR_spon_{area}", f"FR_evk_{area}", f"session_id_{area}",
            ]
            if not all(k in z.files for k in keys):
                continue
            cs = np.asarray(z[f"cvL_spon_{area}"], dtype=float)
            ce = np.asarray(z[f"cvL_evk_{area}"], dtype=float)
            fs = np.asarray(z[f"FR_spon_{area}"], dtype=float)
            fe = np.asarray(z[f"FR_evk_{area}"], dtype=float)
            si = np.asarray(z[f"session_id_{area}"], dtype=int)
            keep = (
                np.isfinite(cs) & np.isfinite(ce) &
                np.isfinite(fs) & np.isfinite(fe) &
                (fs > FR_MIN_HZ) & (fe > FR_MIN_HZ)
            )
            if int(np.sum(keep)) < 20:
                continue
            deltas_s.append(cs[keep])
            deltas_e.append(ce[keep])
            session_ids.append(si[keep])
            used_areas.append(area)
        if deltas_s:
            out.append(summarize(
                f"Mouse {group}\nSteinmetz",
                "Steinmetz mouse",
                np.concatenate(deltas_s),
                np.concatenate(deltas_e),
                "units",
                f"areas={','.join(used_areas)}; matched 0.5 s windows; "
                f"FR>{FR_MIN_HZ:g} Hz in both states; inference on session medians",
                clusters=np.concatenate(session_ids),
            ))
    return out


def load_ibl_groups() -> list[dict]:
    if not IBL_OPS.exists():
        raise FileNotFoundError(IBL_OPS)
    ops = pd.read_csv(IBL_OPS)
    agg = (
        ops.groupby(["eid", "region", "condition"], as_index=False)["cv_L_median"]
        .median(numeric_only=True)
    )
    pv = agg.pivot(index=["eid", "region"], columns="condition", values="cv_L_median")
    pv = pv.dropna(subset=["spontaneous", "active_task"]).reset_index()
    pv["group"] = pv["region"].map(assign_ibl_group)

    out = []
    for group in IBL_GROUP_ORDER:
        sub = pv[pv["group"].eq(group)].copy()
        if len(sub) < 10:
            continue
        out.append(summarize(
            f"Mouse {group}\nIBL/BWM",
            "IBL/BWM mouse",
            sub["spontaneous"].to_numpy(float),
            sub["active_task"].to_numpy(float),
            "region-session pairs",
            f"{sub['region'].nunique()} matched regions; 0.4 s pre-task vs task windows; "
            "inference clustered by recording session",
            clusters=sub["eid"].to_numpy(),
        ))
    return out


def draw(summaries: list[dict]) -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig_h = max(4.2, 0.31 * len(summaries) + 1.1)
    fig, ax = plt.subplots(figsize=(5.8, fig_h))
    y = np.arange(len(summaries))[::-1]
    rng = np.random.default_rng(2)

    for yi, s in zip(y, summaries):
        color = SOURCE_COLORS[s["source"]]
        delta = np.asarray(s["delta"], dtype=float)
        if delta.size:
            if delta.size > 350:
                idx = rng.choice(delta.size, 350, replace=False)
                shown = delta[idx]
            else:
                shown = delta
            jitter = rng.normal(0.0, 0.055, size=len(shown))
            ax.scatter(
                shown,
                np.full_like(shown, yi, dtype=float) + jitter,
                s=7,
                color=color,
                alpha=0.18,
                edgecolor="none",
                rasterized=True,
                zorder=1,
            )

        med = s["delta_median"]
        lo, hi = s["delta_median_ci95"]
        ax.errorbar(
            med, yi,
            xerr=[[med - lo], [hi - med]],
            fmt="D",
            markersize=4.8,
            color="black",
            markerfacecolor=color,
            markeredgecolor="black",
            elinewidth=1.25,
            capsize=3.2,
            zorder=4,
        )
        ax.text(
            1.18, yi,
            (
                f"{p_label(s['inference_wilcoxon_p_two_sided'])}  "
                f"n={s['n']}"
                + (f", s={s['n_clusters']}" if s["n_clusters"] is not None else "")
            ),
            ha="left",
            va="center",
            fontsize=7.3,
        )

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.75, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([s["label"] for s in summaries], fontsize=7.4)
    ax.set_xlabel(r"$\Delta \mathrm{CV}_L$ (baseline $-$ driven)")
    ax.set_title("E  Multi-species and multi-region state contrast", loc="left",
                 fontweight="bold", fontsize=9.6)
    ax.set_xlim(-1.15, 1.55)
    ax.set_ylim(-0.65, len(summaries) - 0.35)
    ax.tick_params(axis="x", labelsize=8)

    # Source legend.
    handles = []
    for source, color in SOURCE_COLORS.items():
        if any(s["source"] == source for s in summaries):
            handles.append(plt.Line2D([0], [0], marker="D", color="black",
                                      markerfacecolor=color, markersize=5,
                                      linestyle="None", label=source))
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    ax.text(
        0.0, -0.13,
        "Positive values indicate higher baseline CV_L.\n"
        "Dots are units or matched region-session pairs; diamonds are pooled medians with 95% CIs.\n"
        "Steinmetz and IBL inference is clustered by session.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
    )

    fig.tight_layout(pad=0.8)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summaries = [load_macaque(), load_allen()]
    summaries.extend(load_steinmetz_groups())
    summaries.extend(load_ibl_groups())
    draw(summaries)

    json_ready = {
        "figure": "Figure2E_cross_dataset_CVL",
        "delta_definition": "spontaneous/pre-task CV_L - evoked/active-task CV_L",
        "excluded_from_state_contrast": {
            "cat_pvc8": "evoked-only; no spontaneous baseline",
            "macaque_pvc11": "spontaneous and evoked recordings are unpaired sessions",
        },
        "rows": [
            {k: v for k, v in s.items() if k != "delta"}
            for s in summaries
        ],
    }
    OUT_JSON.write_text(json.dumps(json_ready, indent=2) + "\n")

    if UPDATED_FIGURES.exists():
        shutil.copy2(OUT_PDF, UPDATED_FIGURES / OUT_PDF.name)
        shutil.copy2(OUT_PNG, UPDATED_FIGURES / OUT_PNG.name)

    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_JSON}")
    for s in summaries:
        print(
            f"{s['label'].replace(chr(10), ' ')}: "
            f"median delta={s['delta_median']:+.3f} "
            f"CI95=[{s['delta_median_ci95'][0]:+.3f}, {s['delta_median_ci95'][1]:+.3f}] "
            f"n={s['n']} clusters={s['n_clusters']} p={s['inference_wilcoxon_p_two_sided']:.3g}"
        )


if __name__ == "__main__":
    main()
