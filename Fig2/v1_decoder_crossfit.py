"""Repeated split-half cross-fitting for the V1 CV--decoder analysis.

For every split, response-dispersion predictors are estimated only from one
half of the trials and decoder accuracy is evaluated on the disjoint half.
The roles of the halves are then exchanged.  Repeated splits stabilize each
session estimate, but recording session remains the inferential unit.

Outputs
-------
Fig2/v1_decoder_crossfit.json
Fig2/v1_decoder_crossfit.npz
outputs/final_text_review_20260907/v1_corrected/Figure_decoder_crossfit.pdf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy
from scipy.stats import pearsonr, rankdata, spearmanr, t as student_t
from sklearn.model_selection import KFold


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_JSON = HERE / "v1_decoder_crossfit.json"
OUT_NPZ = HERE / "v1_decoder_crossfit.npz"
REVIEW_DIR = ROOT / "outputs" / "final_text_review_20260907" / "v1_corrected"
OUT_FIG = REVIEW_DIR / "Figure_decoder_crossfit.pdf"

SESSIONS = [
    "105l001p16.mat",
    "106r001p26.mat",
    "106r002p70.mat",
    "107l002p67.mat",
    "107l003p143.mat",
]
DRIVEN_BINS = 1280
SEED = 20260802
N_REPEATS = 30


def load_v1_driven(path: Path) -> tuple[np.ndarray, np.ndarray]:
    m = sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)
    nd = m["neuralData"][0, 0]
    rasters = nd.spikeRasters
    stim = nd.stim.ravel().astype(int)
    rows, labels = [], []
    for i in range(rasters.shape[0]):
        raster = rasters[i, 0]
        if stim[i] == 0 or raster.shape[1] != DRIVEN_BINS:
            continue
        rows.append(np.asarray(raster.sum(axis=1)).ravel())
        labels.append(stim[i])
    return np.asarray(rows, float), np.asarray(labels, int)


def stratified_half_split(y: np.ndarray, rng: np.random.Generator):
    left, right = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        idx = rng.permutation(idx)
        cut = idx.size // 2
        left.extend(idx[:cut])
        right.extend(idx[cut:])
    return np.asarray(left, int), np.asarray(right, int)


def orientation_degrees(session: str) -> np.ndarray:
    # The session-specific CRCNS table documents a 180-degree stimulus at ID 7
    # for 105l001p16; all other sessions use 135 degrees at that ID.
    if session.startswith("105l001p16"):
        return np.asarray([0, 22.5, 45, 67.5, 90, 112.5, 180, 157.5])
    return np.arange(8, dtype=float) * 22.5


def training_metrics(X: np.ndarray, y: np.ndarray, session: str) -> dict[str, np.ndarray]:
    labels = np.unique(y)
    mean = X.mean(axis=0)
    safe_mean = np.maximum(mean, 1e-12)
    class_means = np.vstack([X[y == label].mean(axis=0) for label in labels])
    within_vars = np.vstack([X[y == label].var(axis=0, ddof=1) for label in labels])

    total_cv = X.std(axis=0, ddof=1) / safe_mean
    tuning_cv = class_means.std(axis=0, ddof=1) / np.maximum(class_means.mean(axis=0), 1e-12)
    noise_cv = np.sqrt(np.nanmean(within_vars, axis=0)) / safe_mean
    modulation_depth = (class_means.max(axis=0) - class_means.min(axis=0)) / safe_mean
    snr = class_means.std(axis=0, ddof=1) / np.sqrt(np.maximum(np.nanmean(within_vars, axis=0), 1e-12))

    theta = np.deg2rad(orientation_degrees(session))[:, None]
    osi = np.abs(np.sum(class_means * np.exp(2j * theta), axis=0)) / np.maximum(
        class_means.sum(axis=0), 1e-12
    )
    return {
        "total_cv": total_cv,
        "tuning_cv": tuning_cv,
        "noise_cv": noise_cv,
        "fr": mean / 1.28,
        "modulation_depth": modulation_depth,
        "snr": snr,
        "osi": osi,
        "class_means": class_means,
        "labels": labels,
    }


def independent_lda_accuracy(
    class_means: np.ndarray,
    labels: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """One-dimensional equal-prior LDA accuracy for every neuron.

    With one feature, balanced classes, and a shared within-class variance, LDA
    assigns each observation to the nearest training-set class mean.  The
    vectorized implementation avoids thousands of equivalent sklearn fits.
    """
    distances = (X_test[:, None, :] - class_means[None, :, :]) ** 2
    pred = labels[np.argmin(distances, axis=1)]
    return np.mean(pred == y_test[:, None], axis=0)


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    """Standard partial Spearman: Pearson correlation of linear rank residuals."""
    matrix = np.column_stack([x, y, *controls]).astype(float)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if matrix.shape[0] <= matrix.shape[1] + 2:
        return float("nan")
    ranks = np.column_stack([rankdata(matrix[:, j]) for j in range(matrix.shape[1])])
    design = np.column_stack([np.ones(ranks.shape[0]), ranks[:, 2:]])
    rx = ranks[:, 0] - design @ np.linalg.lstsq(design, ranks[:, 0], rcond=None)[0]
    ry = ranks[:, 1] - design @ np.linalg.lstsq(design, ranks[:, 1], rcond=None)[0]
    return float(pearsonr(rx, ry).statistic)


def training_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mid-distribution percentiles under the training empirical distribution.

    Ties receive their training probability midpoint; observations outside the
    training range map to 0 or 1. No held-out value changes the reference CDF.
    """
    ordered = np.sort(np.asarray(reference, float))
    if ordered.size == 0 or not np.all(np.isfinite(ordered)):
        raise ValueError("training reference must be nonempty and finite")
    return (
        np.searchsorted(ordered, values, side="left")
        + np.searchsorted(ordered, values, side="right")
    ) / (2.0 * ordered.size)


def rank_fold_prediction(train_values: np.ndarray, test_predictors: np.ndarray):
    ranked = np.column_stack([
        training_percentiles(train_values[:, j], train_values[:, j])
        for j in range(train_values.shape[1])
    ])
    train_design = np.column_stack([np.ones(len(ranked)), ranked[:, 1:]])
    test_design = np.column_stack([
        np.ones(len(test_predictors)),
        *[training_percentiles(train_values[:, j + 1], test_predictors[:, j])
          for j in range(test_predictors.shape[1])],
    ])
    beta = np.linalg.lstsq(train_design, ranked[:, 0], rcond=None)[0]
    return test_design @ beta, float(ranked[:, 0].mean())


def rank_cv_r2(y: np.ndarray, predictors: list[np.ndarray], folds: np.ndarray) -> float:
    values = np.column_stack([y, *predictors]).astype(float)
    keep = np.all(np.isfinite(values), axis=1)
    values, folds = values[keep], folds[keep]
    if values.shape[0] < 20:
        return float("nan")
    outcome = np.full(values.shape[0], np.nan)
    pred = np.full(values.shape[0], np.nan)
    null_pred = np.full(values.shape[0], np.nan)
    for fold in np.unique(folds):
        train = folds != fold
        test = folds == fold
        pred[test], null_pred[test] = rank_fold_prediction(values[train], values[test, 1:])
        # Test outcomes enter only scoring, through the already-fixed training CDF.
        outcome[test] = training_percentiles(values[train, 0], values[test, 0])
    denom = np.sum((outcome - null_pred) ** 2)
    return float(1 - np.sum((outcome - pred) ** 2) / denom) if denom > 0 else float("nan")


def split_statistics(metrics: dict[str, np.ndarray], accuracy: np.ndarray, fold_ids: np.ndarray) -> dict:
    total = metrics["total_cv"]
    tuning = metrics["tuning_cv"]
    noise = metrics["noise_cv"]
    fr = metrics["fr"]
    modulation = metrics["modulation_depth"]
    osi = metrics["osi"]
    snr = metrics["snr"]
    baseline = [fr, modulation, osi, snr]
    r2_base = rank_cv_r2(accuracy, baseline, fold_ids)
    r2_full = rank_cv_r2(accuracy, [*baseline, tuning], fold_ids)
    return {
        "total_cv_accuracy_given_fr": partial_spearman(total, accuracy, [fr]),
        "tuning_cv_accuracy_given_fr": partial_spearman(tuning, accuracy, [fr]),
        "noise_cv_accuracy_given_fr": partial_spearman(noise, accuracy, [fr]),
        "tuning_incremental_given_fr_modulation_osi_snr": partial_spearman(
            tuning, accuracy, baseline
        ),
        "baseline_rank_cv_r2": r2_base,
        "full_rank_cv_r2": r2_full,
        "delta_rank_cv_r2_from_tuning_cv": r2_full - r2_base,
    }


def summarize_session(values: list[dict]) -> dict:
    keys = values[0].keys()
    return {
        key: {
            "median_across_repeats": float(np.nanmedian([v[key] for v in values])),
            "q025": float(np.nanquantile([v[key] for v in values], 0.025)),
            "q975": float(np.nanquantile([v[key] for v in values], 0.975)),
        }
        for key in keys
    }


def session_level_summary(per_session: list[dict], metric: str) -> dict:
    r = np.asarray([s["summary"][metric]["median_across_repeats"] for s in per_session], float)
    z = np.arctanh(np.clip(r, -1 + 1e-9, 1 - 1e-9))
    n = r.size
    mean_z = float(z.mean())
    se_z = float(z.std(ddof=1) / np.sqrt(n))
    crit = float(student_t.ppf(0.975, n - 1))
    t_stat = mean_z / se_z if se_z > 0 else float("inf")
    return {
        "per_session": r.tolist(),
        "median": float(np.median(r)),
        "fisher_mean": float(np.tanh(mean_z)),
        "ci95_cluster_t": np.tanh([mean_z - crit * se_z, mean_z + crit * se_z]).tolist(),
        "t_statistic": float(t_stat),
        "df": int(n - 1),
        "p_two_sided": float(2 * student_t.sf(abs(t_stat), n - 1)),
        "sessions_positive": int(np.sum(r > 0)),
    }


def animal_group_summary(per_session: list[dict], metric: str) -> dict:
    grouped = {}
    for prefix in sorted({row["session_code_group"] for row in per_session}):
        values = [
            row["summary"][metric]["median_across_repeats"]
            for row in per_session if row["session_code_group"] == prefix
        ]
        grouped[prefix] = float(np.mean(values))
    return {
        "mean_session_effect_within_group": grouped,
        "groups_positive": int(np.sum(np.asarray(list(grouped.values())) > 0)),
        "n_documented_animals": 3,
        "scope": "directional summary only; three animals are too few for a stable animal-level variance estimate",
    }


def make_figure(plot_rows: list[dict], session_summaries: list[dict]) -> None:
    total = np.concatenate([row["total_cv"] for row in plot_rows])
    tuning = np.concatenate([row["tuning_cv"] for row in plot_rows])
    noise = np.concatenate([row["noise_cv"] for row in plot_rows])
    acc = np.concatenate([row["accuracy"] for row in plot_rows])
    fr = np.concatenate([row["fr"] for row in plot_rows])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(total, acc, c=np.log10(np.maximum(fr, 1e-3)), s=9, alpha=0.55, cmap="viridis")
    ax.axhline(1 / 8, color="0.45", ls=":", lw=1)
    ax.set_xlabel(r"training-half total $\mathrm{CV}_L$")
    ax.set_ylabel("independent-half decoder accuracy")
    ax.set_title("A  Cross-fitted V1 decoding")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$ firing rate")

    ax = axes[1]
    metrics = [
        "total_cv_accuracy_given_fr",
        "tuning_cv_accuracy_given_fr",
        "noise_cv_accuracy_given_fr",
        "tuning_incremental_given_fr_modulation_osi_snr",
    ]
    labels = [
        r"total CV $\mid$ FR",
        r"tuning CV $\mid$ FR",
        r"noise CV $\mid$ FR",
        r"tuning CV $\mid$ FR, MD, OSI, SNR",
    ]
    vals = np.asarray([
        [s["summary"][m]["median_across_repeats"] for s in session_summaries]
        for m in metrics
    ])
    for i in range(vals.shape[0]):
        ax.scatter(vals[i], np.full(vals.shape[1], i), color="#356a9a", s=27, zorder=3)
        ax.plot([vals[i].min(), vals[i].max()], [i, i], color="#9fb7ce", lw=2, zorder=2)
        fisher_mean = session_level_summary(session_summaries, metrics[i])["fisher_mean"]
        ax.scatter(fisher_mean, i, marker="D", color="#c43c39", s=36, zorder=4)
    ax.axvline(0, color="0.35", lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"session-level partial Spearman $\rho$")
    ax.set_title("B  Five-session cross-fitted effects")
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(n_repeats: int) -> None:
    started = time.perf_counter()
    if n_repeats < 1:
        raise ValueError("at least one repeat is required")
    rng = np.random.default_rng(SEED)
    per_session, plot_rows = [], []
    npz_payload: dict[str, np.ndarray] = {}
    for session_name in SESSIONS:
        X, y = load_v1_driven(HERE / session_name)
        n_neurons = X.shape[1]
        neuron_folds = np.empty(n_neurons, int)
        kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold, (_, test) in enumerate(kfold.split(np.arange(n_neurons))):
            neuron_folds[test] = fold

        repeat_stats = []
        plot_acc, plot_metrics = [], []
        for _ in range(n_repeats):
            left, right = stratified_half_split(y, rng)
            directions = []
            for train_idx, test_idx in ((left, right), (right, left)):
                metrics = training_metrics(X[train_idx], y[train_idx], session_name)
                acc = independent_lda_accuracy(
                    metrics["class_means"], metrics["labels"], X[test_idx], y[test_idx]
                )
                directions.append(split_statistics(metrics, acc, neuron_folds))
                plot_acc.append(acc)
                plot_metrics.append(metrics)
            repeat_stats.append({
                key: float(np.nanmean([directions[0][key], directions[1][key]]))
                for key in directions[0]
            })

        averaged_metrics = {
            key: np.nanmean(np.vstack([m[key] for m in plot_metrics]), axis=0)
            for key in ("total_cv", "tuning_cv", "noise_cv", "fr")
        }
        averaged_accuracy = np.nanmean(np.vstack(plot_acc), axis=0)
        plot_rows.append({**averaged_metrics, "accuracy": averaged_accuracy})
        summary = summarize_session(repeat_stats)
        per_session.append({
            "session": session_name.replace(".mat", ""),
            "session_code_group": session_name[:3],
            "n_trials": int(X.shape[0]),
            "n_neurons": int(n_neurons),
            "summary": summary,
        })
        prefix = session_name.replace(".mat", "")
        for key, value in {**averaged_metrics, "accuracy": averaged_accuracy}.items():
            npz_payload[f"{prefix}_{key}"] = value
        for key in ("total_cv", "tuning_cv", "noise_cv", "fr", "modulation_depth", "osi", "snr"):
            npz_payload[f"{prefix}_split_{key}"] = np.vstack([m[key] for m in plot_metrics])
        npz_payload[f"{prefix}_split_accuracy"] = np.vstack(plot_acc)
        npz_payload[f"{prefix}_neuron_folds"] = neuron_folds
        for key in repeat_stats[0]:
            npz_payload[f"{prefix}_repeat_{key}"] = np.asarray([row[key] for row in repeat_stats])
        print(session_name, json.dumps({k: v["median_across_repeats"] for k, v in summary.items()}, indent=2))

    metrics = [
        "total_cv_accuracy_given_fr",
        "tuning_cv_accuracy_given_fr",
        "noise_cv_accuracy_given_fr",
        "tuning_incremental_given_fr_modulation_osi_snr",
    ]
    output = {
        "analysis_version": "20260907_standard_partial_spearman_train_only_rank_r2",
        "seed": SEED,
        "n_repeats": n_repeats,
        "split_design": "stratified 50/50; metrics on training half, LDA accuracy on disjoint half; halves exchanged",
        "independent_unit": "recording session; repeated splits stabilize estimates and do not add degrees of freedom",
        "session_inference_assumption": "session-level t reference assumes independent session effects; not animal-level inference",
        "partial_correlation_definition": "Pearson correlation of residuals after regressing ranked x and ranked y on intercept and ranked controls",
        "rank_r2_definition": (
            "Five neuron folds; predictor and outcome mid-distribution CDFs are fitted to training neurons only. "
            "Held-out outcomes enter scoring only. R2=1-SSE/SSE_null, with each fold's training-mean "
            "outcome percentile as null prediction. Unclipped linear predictions; identical folds for baseline and full model."
        ),
        "split_cache_layout": "rows alternate left-to-right and right-to-left for each of the 30 repeats; neuron columns preserve raw raster order",
        "dataset_structure": (
            "the public dataset documents five sessions from three animals; the 105, 106, and 107 "
            "session-code groups are retained for a directional animal-level summary"
        ),
        "per_session": per_session,
        "session_level": {metric: session_level_summary(per_session, metric) for metric in metrics},
        "animal_group_level": {metric: animal_group_summary(per_session, metric) for metric in metrics},
        "incremental_rank_r2": {
            "per_session_delta": [
                s["summary"]["delta_rank_cv_r2_from_tuning_cv"]["median_across_repeats"]
                for s in per_session
            ]
        },
        "provenance": {
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "raw_input_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SESSIONS},
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "entropy_recomputed": False,
            "decoder_rule_changed": False,
        },
    }
    if not all(np.all(np.isfinite(value)) for value in npz_payload.values()):
        raise ValueError("nonfinite values in decoder cache")
    np.savez_compressed(OUT_NPZ, **npz_payload)
    output["provenance"]["npz_sha256"] = hashlib.sha256(OUT_NPZ.read_bytes()).hexdigest()
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    OUT_JSON.write_text(serialized, encoding="utf-8")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / OUT_JSON.name).write_text(serialized, encoding="utf-8")
    (REVIEW_DIR / OUT_NPZ.name).write_bytes(OUT_NPZ.read_bytes())
    make_figure(plot_rows, per_session)
    print("Wrote", OUT_JSON, OUT_NPZ, OUT_FIG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    main(args.repeats)
