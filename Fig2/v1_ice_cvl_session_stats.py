"""Session-level inference for the V1 ICE-CV_L association.

The pooled neuron analysis is useful descriptively but does not make neurons
recorded in the same session independent biological replicates. This script
computes the firing-rate-partial Spearman association separately in each of
the five v1v2-1 sessions and combines Fisher-z transformed correlations with
a one-sample t test using four session-level degrees of freedom. Exact
Wilcoxon and sign tests on the five correlations are retained as coarse
distribution-free sensitivities. Figures are written only to the review folder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.stats import binomtest, pearsonr, rankdata, spearmanr, t as student_t, wilcoxon


HERE = Path(__file__).resolve().parent
INFO_PATH = HERE / "unified_v1_Hrho_ICE.mat"
CV_PATH = HERE / "v1_CVL_perN.mat"
OUT_PATH = HERE / "v1_ice_cvl_session_stats.json"
OUT_NPZ = HERE / "v1_ice_cvl_session_stats.npz"
REVIEW_DIR = HERE.parent / "outputs" / "final_text_review_20260907" / "v1_corrected"


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Standard partial Spearman and descriptive t-reference p (df=n-3)."""
    rx, ry, rz = (rankdata(np.asarray(v, float)) for v in (x, y, z))
    rz_centered = rz - rz.mean()
    denom = float(np.dot(rz_centered, rz_centered))
    if denom <= 0:
        raise ValueError("firing-rate ranks have zero variance")
    rx_centered = rx - rx.mean()
    ry_centered = ry - ry.mean()
    ex = rx_centered - np.dot(rz_centered, rx_centered) / denom * rz_centered
    ey = ry_centered - np.dot(rz_centered, ry_centered) / denom * rz_centered
    rho = float(pearsonr(ex, ey).statistic)
    df = rx.size - 3
    if df <= 0 or not np.isfinite(rho):
        raise ValueError("insufficient observations or degenerate rank residuals")
    statistic = rho * np.sqrt(df / max(1.0 - rho * rho, np.finfo(float).eps))
    return rho, float(2 * student_t.sf(abs(statistic), df))


def summarize_session_correlations(values: list[float]) -> dict:
    r = np.asarray(values, float)
    z = np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12))
    n = int(r.size)
    df = n - 1
    mean_z = float(z.mean())
    se_z = float(z.std(ddof=1) / np.sqrt(n))
    t_stat = mean_z / se_z
    p_t = float(2 * student_t.sf(abs(t_stat), df))
    crit = float(student_t.ppf(0.975, df))
    ci_z = np.asarray([mean_z - crit * se_z, mean_z + crit * se_z])
    n_positive = int(np.sum(r > 0))
    return {
        "n_sessions": n,
        "median_partial_rho": float(np.median(r)),
        "fisher_mean_rho": float(np.tanh(mean_z)),
        "fisher_mean_rho_ci95_cluster_t": np.tanh(ci_z).tolist(),
        "fisher_z_t_statistic": float(t_stat),
        "cluster_df": df,
        "fisher_z_t_p_two_sided": p_t,
        "sessions_with_positive_rho": n_positive,
        "session_wilcoxon_p_two_sided": float(wilcoxon(r).pvalue),
        "session_sign_test_p_two_sided": float(binomtest(n_positive, n, 0.5).pvalue),
    }


def main() -> None:
    info = loadmat(INFO_PATH, squeeze_me=True, struct_as_record=False)
    cv_data = loadmat(CV_PATH, squeeze_me=True, struct_as_record=False)
    sessions = np.ravel(info["results"].v1v2)
    cv_all = np.ravel(cv_data["cvL_drv"]).astype(float)
    session_ids = np.ravel(cv_data["sess_id"]).astype(int)
    raw_fr_all = np.ravel(cv_data["FR_drv"]).astype(float)
    if cv_all.shape != raw_fr_all.shape or cv_all.shape != session_ids.shape:
        raise ValueError("CV, original firing rate, and session IDs are unaligned")

    rows = []
    pooled = {"cv": [], "fr": [], "ice_h": [], "ice_i": []}
    payload = {}
    for session_index, session in enumerate(sessions, start=1):
        hr = np.ravel(session.hr).astype(float)
        mi = np.ravel(session.mi).astype(float)
        name = str(np.ravel(cv_data["V1V2_FILES"])[session_index - 1])
        if str(session.file) != name:
            raise ValueError(f"session identity mismatch: {session.file} versus {name}")
        filtered_fr = np.ravel(session.FR).astype(float)
        fr = raw_fr_all[session_ids == session_index]
        cost = 2.4 + 0.37 * fr
        cv = cv_all[session_ids == session_index]
        if not (hr.size == mi.size == cost.size == fr.size == cv.size == filtered_fr.size):
            raise ValueError(f"unaligned arrays in session {session_index}")

        keep = (
            np.isfinite(hr) & np.isfinite(mi) & np.isfinite(cost) & (cost > 0)
            & np.isfinite(fr) & (fr > 0) & np.isfinite(cv) & (cv > 0)
        )
        ice_h = hr[keep] / cost[keep]
        ice_i = mi[keep] / cost[keep]
        rho_h, p_h = partial_spearman(cv[keep], ice_h, fr[keep])
        rho_i, p_i = partial_spearman(cv[keep], ice_i, fr[keep])
        rows.append({
            "session_index": session_index,
            "session": name,
            "n_neurons": int(keep.sum()),
            "partial_rho_H_over_cost_given_FR": float(rho_h),
            "neuron_level_p_H_descriptive": float(p_h),
            "partial_rho_I_over_cost_given_FR": float(rho_i),
            "neuron_level_p_I_descriptive": float(p_i),
            "max_original_minus_filtered_rate_Hz": float(np.max(fr - filtered_fr)),
        })
        prefix = Path(name).stem
        for key, value in {
            "cv": cv[keep], "fr": fr[keep], "filtered_word_stream_fr": filtered_fr[keep],
            "cost": cost[keep], "hr": hr[keep], "mi": mi[keep],
            "ice_h": ice_h, "ice_i": ice_i, "neuron_index_1based": np.flatnonzero(keep) + 1,
        }.items():
            payload[f"{prefix}_{key}"] = value
        for key, value in (
            ("cv", cv[keep]), ("fr", fr[keep]),
            ("ice_h", ice_h), ("ice_i", ice_i),
        ):
            pooled[key].append(value)

    pooled = {key: np.concatenate(value) for key, value in pooled.items()}
    pooled_h = partial_spearman(pooled["cv"], pooled["ice_h"], pooled["fr"])
    pooled_i = partial_spearman(pooled["cv"], pooled["ice_i"], pooled["fr"])
    output = {
        "analysis_version": "20260907_standard_partial_spearman_original_count_rate",
        "n_sessions": len(rows),
        "n_neurons_pooled": int(pooled["cv"].size),
        "pooled_neuron_level_descriptive": {
            "partial_rho_H_over_cost_given_FR": pooled_h[0],
            "p_H": pooled_h[1],
            "partial_rho_I_over_cost_given_FR": pooled_i[0],
            "p_I": pooled_i[1],
            "raw_rho_H_over_cost": float(spearmanr(pooled["cv"], pooled["ice_h"]).statistic),
            "raw_p_H_over_cost": float(spearmanr(pooled["cv"], pooled["ice_h"]).pvalue),
            "raw_rho_I_over_cost": float(spearmanr(pooled["cv"], pooled["ice_i"]).statistic),
            "raw_p_I_over_cost": float(spearmanr(pooled["cv"], pooled["ice_i"]).pvalue),
        },
        "per_session": rows,
        "session_level_H_over_cost": summarize_session_correlations(
            [row["partial_rho_H_over_cost_given_FR"] for row in rows]
        ),
        "session_level_I_over_cost": summarize_session_correlations(
            [row["partial_rho_I_over_cost_given_FR"] for row in rows]
        ),
        "inference_note": (
            "Recording session is the inferential unit, conditional on independent session effects; "
            "this is not animal-level inference. Pooled neuron p-values "
            "are descriptive; primary model-based inference is the Fisher-z "
            "one-sample t test with n_sessions-1 degrees of freedom. Neuron-level partial-correlation "
            "p-values use an approximate t reference with n-3 degrees of freedom."
        ),
        "partial_correlation_definition": "Pearson correlation of linear rank residuals; no second ranking of residuals",
        "rate_and_cost_definition": "FR_drv from v1_CVL_perN.mat: original trial counts divided by 1.28 s; cost=2.4+0.37*FR_drv",
        "entropy_word_stream_note": (
            "Existing H(W) and I((S,J);W) estimates are retained without recomputation. "
            "Their binary word stream applies a one-following-bin dead time: an accepted spike deletes "
            "a spike in the next 1-ms bin. CV, rate control, and cost instead use original trial counts. "
            "The entropy-cache FR and C are not used as the control or denominator."
        ),
        "provenance": {
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (INFO_PATH, CV_PATH)},
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "entropy_recomputed": False,
        },
    }
    if not all(np.all(np.isfinite(value)) for value in payload.values()):
        raise ValueError("nonfinite values in information-association cache")
    np.savez_compressed(OUT_NPZ, **payload)
    output["provenance"]["npz_sha256"] = hashlib.sha256(OUT_NPZ.read_bytes()).hexdigest()
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    OUT_PATH.write_text(serialized, encoding="utf-8")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / OUT_PATH.name).write_text(serialized, encoding="utf-8")
    (REVIEW_DIR / OUT_NPZ.name).write_bytes(OUT_NPZ.read_bytes())
    make_figure(output, pooled)
    print(json.dumps(output, indent=2))
    print("Wrote", OUT_PATH)


def make_figure(output: dict, pooled: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    scatter = axes[0].scatter(pooled["cv"], pooled["ice_i"],
                              c=np.log10(pooled["fr"]), s=9, alpha=0.55, cmap="viridis")
    axes[0].set(xlabel=r"original-count $\mathrm{CV}_L$", ylabel=r"$I((S,J);W)/\widetilde C$",
                title="A  Original-rate cost normalization")
    fig.colorbar(scatter, ax=axes[0], label=r"$\log_{10}$ original firing rate (Hz)")
    for i, kind in enumerate(("I", "H")):
        summary = output[f"session_level_{kind}_over_cost"]
        values = [row[f"partial_rho_{kind}_over_cost_given_FR"] for row in output["per_session"]]
        axes[1].scatter(values, np.full(len(values), i), s=28, color="#356a9a")
        low, high = summary["fisher_mean_rho_ci95_cluster_t"]
        axes[1].plot([low, high], [i, i], color="#c43c39", lw=2)
        axes[1].scatter(summary["fisher_mean_rho"], i, marker="D", color="#c43c39")
    axes[1].axvline(0, color="0.4", lw=1)
    axes[1].set(yticks=[0, 1], yticklabels=["I/cost | original FR", "H/cost | original FR"],
                xlabel=r"standard partial Spearman $\rho$", title="B  Five-session effects and 95% CIs",
                ylim=(-0.5, 1.5))
    axes[1].invert_yaxis()
    for suffix in ("pdf", "png"):
        fig.savefig(REVIEW_DIR / f"Figure_v1_ice_cvl_session_stats.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
