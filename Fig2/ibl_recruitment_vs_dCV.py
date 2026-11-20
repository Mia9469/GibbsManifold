"""Rate and cross-validated decoder correlates of the CV_L state shift.

Region-session pairs are useful for showing the distribution, but they are
nested within recording sessions. Panels B-C therefore use one median per
session as the independent unit and report session-level permutation tests.
The light region-session points are descriptive only.

Outputs
-------
Fig2/FigureS18_recruitment_vs_dCV.{pdf,png}
updated/figures/FigureS18_recruitment_vs_dCV.{pdf,png}
Fig2/ibl_recruitment_v3_pairs.csv
Fig2/ibl_recruitment_vs_dCV_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IBL_OUTPUTS = REPO.parent / "IBL_data" / "analysis_outputs"
OPS_CSV = IBL_OUTPUTS / "bits_per_spike" / "tier1_operating_points.csv"
GEOMETRY_CSV = IBL_OUTPUTS / "recomputed" / "gm_ops_v3_geometry_balanced.csv"
DECODER_CSV = (
    IBL_OUTPUTS / "regional_mechanisms" / "v7_hpf_choice_block_decoders.csv"
)
PAIR_CACHE = HERE / "ibl_recruitment_v3_pairs.csv"
STEINMETZ_NPZ = HERE / "steinmetz_CVL_results.npz"
OUT_PDF = HERE / "FigureS18_recruitment_vs_dCV.pdf"
OUT_PNG = HERE / "FigureS18_recruitment_vs_dCV.png"
OUT_MANUSCRIPT = REPO / "updated" / "figures"
OUT_JSON = HERE / "ibl_recruitment_vs_dCV_summary.json"

N_PERM = 100_000
N_BOOT = 20_000
PERM_SEED = 20260905
BOOT_SEED = 20260906

STEINMETZ_GROUPS = {
    "Visual": ["VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa"],
    "Thalamus": ["TH", "LGd", "LP", "MD", "PO", "POL", "LD", "VPL", "MG"],
    "Frontal/PFC": ["MOs", "ACA", "PL", "ILA", "ORBm"],
    "Striatum/BG": ["CP", "GPe"],
    "Hippocampus": ["CA1", "CA3", "DG", "SUB", "POST"],
}
GROUP_COLORS = {
    "Visual": "#4779C7",
    "Thalamus": "#8E6FB0",
    "Frontal/PFC": "#C0392B",
    "Striatum/BG": "#2E9E6B",
    "Hippocampus": "#D98734",
}


def _finite_rows(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    values = [np.asarray(a, dtype=float).ravel() for a in arrays]
    keep = np.logical_and.reduce([np.isfinite(a) for a in values])
    return tuple(a[keep] for a in values)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation after linear residualization of ranks on z."""
    x, y, z = _finite_rows(x, y, z)
    if len(x) < 4:
        return float("nan")
    rx, ry, rz = (rankdata(a).astype(float) for a in (x, y, z))
    zc = rz - rz.mean()
    zss = float(np.dot(zc, zc))
    if zss <= 0:
        return float("nan")

    def residual(r: np.ndarray) -> np.ndarray:
        rc = r - r.mean()
        return rc - np.dot(rc, zc) / zss * zc

    ex, ey = residual(rx), residual(ry)
    denom = np.sqrt(np.dot(ex, ex) * np.dot(ey, ey))
    return float(np.dot(ex, ey) / denom) if denom > 0 else float("nan")


def permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    *,
    seed: int,
    n_perm: int = N_PERM,
) -> tuple[float, float]:
    """Two-sided session-label permutation test for raw or partial Spearman rho."""
    if z is None:
        x, y = _finite_rows(x, y)
        rx, ry = (rankdata(a).astype(float) for a in (x, y))
        ex = rx - rx.mean()
        yc = ry - ry.mean()
        obs = float(np.dot(ex, yc) / np.sqrt(np.dot(ex, ex) * np.dot(yc, yc)))
        zc = None
        zss = None
    else:
        x, y, z = _finite_rows(x, y, z)
        rx, ry, rz = (rankdata(a).astype(float) for a in (x, y, z))
        zc = rz - rz.mean()
        zss = float(np.dot(zc, zc))
        xc = rx - rx.mean()
        yc = ry - ry.mean()
        ex = xc - np.dot(xc, zc) / zss * zc
        ey = yc - np.dot(yc, zc) / zss * zc
        obs = float(np.dot(ex, ey) / np.sqrt(np.dot(ex, ex) * np.dot(ey, ey)))

    rng = np.random.default_rng(seed)
    exss = float(np.dot(ex, ex))
    extreme = 0
    batch_size = 5000
    for start in range(0, n_perm, batch_size):
        batch = min(batch_size, n_perm - start)
        # Sorting independent continuous draws gives uniform random permutations.
        indices = np.argsort(rng.random((batch, len(yc))), axis=1)
        y_perm = yc[indices]
        numer = np.einsum("ij,j->i", y_perm, ex)
        if zc is None:
            yss = np.einsum("ij,ij->i", y_perm, y_perm)
        else:
            yz = np.einsum("ij,j->i", y_perm, zc)
            yss = np.einsum("ij,ij->i", y_perm, y_perm) - yz**2 / zss
        null = numer / np.sqrt(exss * np.maximum(yss, np.finfo(float).tiny))
        extreme += int(np.count_nonzero(np.abs(null) >= abs(obs) - 1e-15))
    p = (extreme + 1) / (n_perm + 1)
    return obs, float(p)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    *,
    seed: int,
    n_boot: int = N_BOOT,
) -> list[float]:
    """Percentile bootstrap CI after resampling independent sessions."""
    arrays = _finite_rows(x, y) if z is None else _finite_rows(x, y, z)
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    estimates = np.full(n_boot, np.nan, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if z is None:
            estimates[i] = spearmanr(arrays[0][idx], arrays[1][idx]).statistic
        else:
            estimates[i] = partial_spearman(
                arrays[0][idx], arrays[1][idx], arrays[2][idx]
            )
    estimates = estimates[np.isfinite(estimates)]
    return [float(v) for v in np.percentile(estimates, [2.5, 97.5])]


def load_steinmetz_mechanism() -> dict:
    """Per-neuron baseline-task CV_L and firing-rate shifts by broad region."""
    data = np.load(STEINMETZ_NPZ, allow_pickle=True)
    suffixes = sorted(
        {key.split("_", 2)[2] for key in data.files if key.startswith("cvL_spon_")}
    )
    area_to_group = {
        area: group for group, areas in STEINMETZ_GROUPS.items() for area in areas
    }

    dcv, dfr, groups = [], [], []
    for area in suffixes:
        cv_spon, cv_active = data[f"cvL_spon_{area}"], data[f"cvL_evk_{area}"]
        fr_spon, fr_active = data[f"FR_spon_{area}"], data[f"FR_evk_{area}"]
        keep = (
            np.isfinite(cv_spon)
            & np.isfinite(cv_active)
            & np.isfinite(fr_spon)
            & np.isfinite(fr_active)
        )
        dcv.append((cv_spon - cv_active)[keep])
        dfr.append((fr_active - fr_spon)[keep])
        groups.append(
            np.full(int(keep.sum()), area_to_group.get(area, "Other"), dtype=object)
        )
    dcv = np.concatenate(dcv)
    dfr = np.concatenate(dfr)
    groups = np.concatenate(groups)
    rho, p = spearmanr(dcv, dfr)

    by_group = {}
    for group in list(STEINMETZ_GROUPS) + ["Other"]:
        keep = groups == group
        if keep.sum() == 0:
            continue
        by_group[group] = {
            "n": int(keep.sum()),
            "median_dFR_active_minus_base": float(np.median(dfr[keep])),
            "median_dCV_base_minus_active": float(np.median(dcv[keep])),
        }
    return {
        "arrays": {"dCV": dcv, "dFR": dfr, "group": groups},
        "summary": {
            "n_neurons": int(len(dcv)),
            "spearman_dCV_vs_dFR": float(rho),
            "spearman_p": float(p),
            "inference_unit": "neuron-level descriptive; neurons are nested within sessions",
            "by_group": by_group,
        },
    }


def build_pairs() -> pd.DataFrame:
    """Build matched v3 region-session pairs, falling back to the local cache."""
    source_paths = (OPS_CSV, GEOMETRY_CSV, DECODER_CSV)
    if not all(path.exists() for path in source_paths):
        if not PAIR_CACHE.exists():
            missing = ", ".join(str(path) for path in source_paths if not path.exists())
            raise FileNotFoundError(f"Missing IBL source tables and no cache: {missing}")
        return pd.read_csv(PAIR_CACHE)

    operating = pd.read_csv(OPS_CSV)
    operating = operating[operating["condition"].isin(["active_task", "spontaneous"])]
    operating = (
        operating.groupby(["eid", "region", "condition"], as_index=False)[
            ["cv_L_median", "FR_median_hz"]
        ]
        .median()
    )
    wide = operating.pivot(
        index=["eid", "region"],
        columns="condition",
        values=["cv_L_median", "FR_median_hz"],
    )
    wide.columns = [f"{metric}__{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index().dropna(
        subset=[
            "cv_L_median__active_task",
            "cv_L_median__spontaneous",
            "FR_median_hz__active_task",
            "FR_median_hz__spontaneous",
        ]
    )
    wide["dCV"] = (
        wide["cv_L_median__spontaneous"] - wide["cv_L_median__active_task"]
    )
    wide["dFR"] = (
        wide["FR_median_hz__active_task"] - wide["FR_median_hz__spontaneous"]
    )

    geometry = (
        pd.read_csv(GEOMETRY_CSV)
        .groupby(["eid", "region"], as_index=False)[
            ["decoder_stim_above_chance"]
        ]
        .median()
    )
    choice = (
        pd.read_csv(DECODER_CSV)
        .groupby(["eid", "region"], as_index=False)["decoder_choice_above_chance"]
        .median()
    )
    pairs = wide.merge(geometry, on=["eid", "region"], how="left")
    pairs = pairs.merge(choice, on=["eid", "region"], how="left")
    pairs.to_csv(PAIR_CACHE, index=False)
    return pairs


def summarize_metric(
    pairs: pd.DataFrame, metric: str, *, seed_offset: int
) -> tuple[dict, pd.DataFrame]:
    columns = ["eid", "region", "dCV", "dFR", metric]
    pair_rows = pairs[columns].replace([np.inf, -np.inf], np.nan).dropna()
    sessions = pair_rows.groupby("eid", as_index=False).median(numeric_only=True)

    pair_raw = spearmanr(pair_rows["dCV"], pair_rows[metric]).statistic
    pair_partial = partial_spearman(
        pair_rows["dCV"], pair_rows[metric], pair_rows["dFR"]
    )
    raw_rho, raw_p = permutation_p(
        sessions["dCV"].to_numpy(),
        sessions[metric].to_numpy(),
        None,
        seed=PERM_SEED + seed_offset,
    )
    partial_rho, partial_p = permutation_p(
        sessions["dCV"].to_numpy(),
        sessions[metric].to_numpy(),
        sessions["dFR"].to_numpy(),
        seed=PERM_SEED + 100 + seed_offset,
    )
    summary = {
        "n_region_session_pairs": int(len(pair_rows)),
        "n_sessions": int(len(sessions)),
        "pair_level_descriptive": {
            "spearman_rho": float(pair_raw),
            "spearman_rho_partial_FR": float(pair_partial),
            "note": "Region-session pairs are nested and are not inferential replicates.",
        },
        "session_level": {
            "aggregation": "median across available regions within each session",
            "spearman_rho": raw_rho,
            "spearman_permutation_p_two_sided": raw_p,
            "spearman_ci95_session_bootstrap": bootstrap_ci(
                sessions["dCV"].to_numpy(),
                sessions[metric].to_numpy(),
                None,
                seed=BOOT_SEED + seed_offset,
            ),
            "spearman_rho_partial_FR": partial_rho,
            "spearman_partial_FR_permutation_p_two_sided": partial_p,
            "spearman_partial_FR_ci95_session_bootstrap": bootstrap_ci(
                sessions["dCV"].to_numpy(),
                sessions[metric].to_numpy(),
                sessions["dFR"].to_numpy(),
                seed=BOOT_SEED + 100 + seed_offset,
            ),
            "n_permutations": N_PERM,
            "n_bootstrap": N_BOOT,
        },
    }
    return summary, sessions


def format_p(p: float) -> str:
    if p < 1e-4:
        return r"$p_{\rm perm}<10^{-4}$"
    if p < 0.1:
        return rf"$p_{{\rm perm}}={p:.3f}$"
    return rf"$p_{{\rm perm}}={p:.2f}$"


def main() -> None:
    pairs = build_pairs()
    metrics = [
        (
            "decoder_choice_above_chance",
            "Choice decodability\n(accuracy above chance)",
        ),
        (
            "decoder_stim_above_chance",
            "Stimulus decodability\n(accuracy above chance)",
        ),
    ]
    results = {}
    session_tables = {}
    for offset, (metric, _) in enumerate(metrics):
        results[metric], session_tables[metric] = summarize_metric(
            pairs, metric, seed_offset=offset
        )

    mechanism = load_steinmetz_mechanism()
    summary = {
        "dCV_definition": (
            "cv_L(spontaneous/baseline) - cv_L(active_task); positive values mean "
            "higher baseline CV_L"
        ),
        "panelA_steinmetz_rate_mechanism": {
            "source": str(STEINMETZ_NPZ),
            **mechanism["summary"],
        },
        "panelsBC_ibl_session_level_controls": {
            "source_tables": [str(OPS_CSV), str(GEOMETRY_CSV), str(DECODER_CSV)],
            "local_pair_cache": str(PAIR_CACHE),
            "n_matched_pairs_total": int(len(pairs)),
            "inference": (
                "Region-session pairs are shown descriptively. Primary correlations, "
                "FR-partial correlations, permutation p-values, and bootstrap intervals "
                "use one median per recording session."
            ),
            "correlations": results,
            "result": (
                "Rate changes account for much of the cross-region CV_L shift. At the "
                "session level, neither choice nor stimulus decodability provides a "
                "uniform positive counterpart to the shift after firing-rate control."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.65))

    ax = axes[0]
    dcv = mechanism["arrays"]["dCV"]
    dfr = mechanism["arrays"]["dFR"]
    rng = np.random.default_rng(0)
    shown = rng.choice(len(dcv), size=min(4000, len(dcv)), replace=False)
    ax.scatter(
        dfr[shown],
        dcv[shown],
        s=4,
        color="#BBBBBB",
        alpha=0.25,
        edgecolor="none",
        rasterized=True,
        zorder=1,
    )
    for group, color in GROUP_COLORS.items():
        stats = mechanism["summary"]["by_group"].get(group)
        if stats is None:
            continue
        ax.scatter(
            stats["median_dFR_active_minus_base"],
            stats["median_dCV_base_minus_active"],
            s=75,
            marker="D",
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=f"{group} (n={stats['n']})",
            zorder=4,
        )
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r"$\Delta\bar r$ (active $-$ baseline, Hz)")
    ax.set_ylabel(r"$\Delta\mathrm{CV}_L$ (baseline $-$ active)")
    ax.text(-0.13, 1.03, "A", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.text(
        0.03,
        0.97,
        r"Neuron-level descriptive: $\rho=+0.71$" + "\n" + r"$p<10^{-300}$; $n=10{,}163$",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.legend(frameon=False, fontsize=6.3, loc="lower right")

    for ax, panel, (metric, xlabel) in zip(axes[1:], "BC", metrics):
        pair_rows = pairs[["eid", "dCV", metric]].dropna()
        sessions = session_tables[metric]
        ax.scatter(
            pair_rows[metric],
            pair_rows["dCV"],
            s=15,
            color="#9A7BB5",
            alpha=0.22,
            edgecolor="none",
            label="Region-session pairs",
            zorder=1,
        )
        ax.scatter(
            sessions[metric],
            sessions["dCV"],
            s=35,
            marker="D",
            color="black",
            edgecolor="white",
            linewidth=0.35,
            label="Session medians",
            zorder=3,
        )
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta\mathrm{CV}_L$ (baseline $-$ task)")
        ax.text(-0.13, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
        session = results[metric]["session_level"]
        ax.text(
            0.03,
            0.97,
            "Session medians, FR-controlled\n"
            + rf"$\rho={session['spearman_rho_partial_FR']:+.2f}$, "
            + format_p(session["spearman_partial_FR_permutation_p_two_sided"])
            + rf"; $n={results[metric]['n_sessions']}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(w_pad=1.4)

    OUT_MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    for path in (OUT_PDF, OUT_MANUSCRIPT / OUT_PDF.name):
        fig.savefig(path, bbox_inches="tight")
    for path in (OUT_PNG, OUT_MANUSCRIPT / OUT_PNG.name):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_PDF}\nSaved {OUT_PNG}\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
