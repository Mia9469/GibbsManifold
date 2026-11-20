"""Native-count, shared-shape leave-one-session-out candidate comparison.

Each candidate has one response-shape parameter shared across the four training
sessions and one neuron-specific rate/scale parameter profiled by native-count
likelihood on a calibration half of the held-out neuron's trials.  Log
predictive density is then evaluated on the disjoint half.  Thus repeated trial
splits stabilize the held-out-session score without changing the five-session
replication unit.

Candidates
----------
* power-cost maximum entropy: p(k) proportional to exp(-beta k**alpha)
* discretized lognormal with shared sigma
* discretized gamma with shared shape
* negative binomial with shared size/dispersion

Outputs
-------
Fig5/shared_shape_count_loso.json
Fig5/shared_shape_count_loso.npz
outputs/final_text_review_20260907/v1_corrected/Figure_shared_shape_count_loso.pdf
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import gammainc, gammaincc, gammaln, logsumexp, ndtr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN_SCRIPT = HERE / "discriminative_maxent_test.py"
OUT_JSON = HERE / "shared_shape_count_loso.json"
OUT_NPZ = HERE / "shared_shape_count_loso.npz"
REVIEW_DIR = ROOT / "outputs" / "final_text_review_20260907" / "v1_corrected"
OUT_FIG = REVIEW_DIR / "Figure_shared_shape_count_loso.pdf"

SEED = 20260802
N_SPLITS = 8
COUNT_MAX = 1280
GRID = np.arange(COUNT_MAX + 1, dtype=float)
LOWER = np.maximum(GRID - 0.5, 0.0)
UPPER = GRID + 0.5
PROB_FLOOR = 1e-300
LOG_GRID = np.log(np.maximum(GRID, np.finfo(float).tiny))
LOG_GRID[0] = -np.inf
MODELS = ("power_cost", "lognormal", "gamma", "negative_binomial")
DISPLAY = {
    "power_cost": "power-cost",
    "lognormal": "lognormal",
    "gamma": "gamma",
    "negative_binomial": "negative binomial",
}
BOUNDS = {
    "power_cost": (0.20, 8.0),
    "lognormal": (0.05, 3.0),
    "gamma": (0.10, 80.0),
    "negative_binomial": (0.05, 200.0),
}
NUISANCE_AUDIT = {}
NEURON_IDS = {}


def load_main_module():
    spec = importlib.util.spec_from_file_location("discriminative_main", MAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN = load_main_module()


def hist_rows(samples: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(samples), COUNT_MAX + 1), dtype=float)
    for i, values in enumerate(samples):
        x = np.asarray(values, int)
        if np.any((x < 0) | (x > COUNT_MAX)):
            raise ValueError("count outside declared support")
        out[i] = np.bincount(x, minlength=COUNT_MAX + 1)
    return out


def means_from_hist(hist: np.ndarray) -> np.ndarray:
    return np.sum(hist * GRID[None, :], axis=1) / np.maximum(hist.sum(axis=1), 1.0)


def power_logbeta(alpha: float, hist: np.ndarray) -> np.ndarray:
    """Solve the discrete likelihood moment equation on the positive-beta branch.

    Scaling k**alpha by its observed moment makes the target exactly one.
    The search variable is log(beta * observed_moment), with adaptive brackets
    and safeguarded Newton steps, not a fixed beta or mean restriction.
    """
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("power shape must be finite and positive")
    moment = np.sum(hist * np.power(GRID, alpha)[None, :], axis=1) / hist.sum(axis=1)
    if not np.all(np.isfinite(moment) & (moment > 0)):
        raise ValueError("positive finite empirical power moments are required")
    sufficient = np.power(GRID, alpha)[None, :] / moment[:, None]
    if np.any(sufficient.mean(axis=1) <= 1):
        raise ValueError("empirical moment is outside the positive-beta branch; uniform/negative-beta boundary requires explicit treatment")

    def moments(z):
        lam = np.exp(z)
        logw = -lam[:, None] * sufficient
        prob = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
        expectation = np.sum(prob * sufficient, axis=1)
        variance = np.sum(prob * (sufficient - expectation[:, None]) ** 2, axis=1)
        return expectation, lam * variance

    z = np.full(hist.shape[0], -np.log(alpha))
    low, high = z - 4, z + 4
    for _ in range(60):
        need_low = moments(low)[0] <= 1
        need_high = moments(high)[0] >= 1
        if not np.any(need_low | need_high):
            break
        low[need_low] -= 4
        high[need_high] += 4
    else:
        raise RuntimeError("could not bracket positive-beta moment root")
    for _ in range(80):
        expectation, derivative_scale = moments(z)
        error = expectation - 1
        if np.max(np.abs(error)) < 2e-9:
            result = z - np.log(moment)
            record_nuisance_audit("power_cost", result, np.abs(error))
            return result
        low = np.where(error > 0, z, low)
        high = np.where(error < 0, z, high)
        step = np.divide(error, derivative_scale, out=np.zeros_like(error), where=derivative_scale > 0)
        proposed = z + step
        accepted = np.isfinite(proposed) & (proposed > low) & (proposed < high)
        z = np.where(np.abs(error) < 2e-9, z, np.where(accepted, proposed, (low + high) / 2))
    raise RuntimeError("power-cost likelihood profile did not converge")


def power_beta(alpha: float, hist: np.ndarray) -> np.ndarray:
    return np.exp(power_logbeta(alpha, hist))


def power_logpmf_with_logbeta(alpha: float, logbeta: np.ndarray) -> np.ndarray:
    logw = -np.exp(np.asarray(logbeta)[:, None] + alpha * LOG_GRID[None, :])
    return logw - logsumexp(logw, axis=1, keepdims=True)


def power_logpmf_with_beta(alpha: float, beta: np.ndarray) -> np.ndarray:
    return power_logpmf_with_logbeta(alpha, np.log(beta))


def power_logpmf(alpha: float, hist: np.ndarray) -> np.ndarray:
    return power_logpmf_with_logbeta(alpha, power_logbeta(alpha, hist))


def lognormal_logpmf_with_location(sigma: float, location: np.ndarray) -> np.ndarray:
    upper = (np.log(UPPER)[None, :] - location[:, None]) / sigma
    lower = (np.log(np.maximum(LOWER, np.finfo(float).tiny))[None, :] - location[:, None]) / sigma
    lower[:, 0] = -np.inf
    probs = np.where(lower > 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))
    norm = ndtr((np.log(COUNT_MAX + 0.5) - location[:, None]) / sigma)
    probs = probs / np.maximum(norm, PROB_FLOOR)
    probs = np.clip(probs, PROB_FLOOR, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


def lognormal_logpmf(sigma: float, hist: np.ndarray) -> np.ndarray:
    mu = np.maximum(means_from_hist(hist), 1e-3)
    location = np.log(mu) - 0.5 * sigma * sigma
    return lognormal_logpmf_with_location(sigma, location)


def gamma_logpmf_with_scale(shape: float, scale: np.ndarray) -> np.ndarray:
    upper, lower = UPPER[None, :] / scale[:, None], LOWER[None, :] / scale[:, None]
    cdf_lower, cdf_upper = gammainc(shape, lower), gammainc(shape, upper)
    probs = np.where(cdf_lower > 0.5,
                     gammaincc(shape, lower) - gammaincc(shape, upper), cdf_upper - cdf_lower)
    norm = gammainc(shape, (COUNT_MAX + 0.5) / scale[:, None])
    probs = probs / np.maximum(norm, PROB_FLOOR)
    probs = np.clip(probs, PROB_FLOOR, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


def gamma_logpmf(shape: float, hist: np.ndarray) -> np.ndarray:
    mu = np.maximum(means_from_hist(hist), 1e-3)
    scale = np.maximum(mu / shape, 1e-6)
    return gamma_logpmf_with_scale(shape, scale)


def negative_binomial_logpmf_with_mean(size: float, mu: np.ndarray) -> np.ndarray:
    log_denom = np.logaddexp(np.log(size), np.log(mu))
    k = GRID[None, :]
    logp = (
        gammaln(k + size) - gammaln(size) - gammaln(k + 1)
        + size * (np.log(size) - log_denom[:, None])
        + k * (np.log(mu)[:, None] - log_denom[:, None])
    )
    logp -= logsumexp(logp, axis=1, keepdims=True)
    return logp


def negative_binomial_logpmf(size: float, hist: np.ndarray) -> np.ndarray:
    return negative_binomial_logpmf_with_mean(
        size, np.maximum(means_from_hist(hist), 1e-9)
    )


def model_logpmf_moment_calibrated(model: str, shape: float, hist: np.ndarray) -> np.ndarray:
    if model == "power_cost":
        return power_logpmf(shape, hist)
    if model == "lognormal":
        return lognormal_logpmf(shape, hist)
    if model == "gamma":
        return gamma_logpmf(shape, hist)
    if model == "negative_binomial":
        return negative_binomial_logpmf(shape, hist)
    raise KeyError(model)


def model_logpmf_with_nuisance(model: str, shape: float, nuisance: np.ndarray) -> np.ndarray:
    if model == "power_cost":
        return power_logpmf_with_logbeta(shape, nuisance)
    if model == "lognormal":
        return lognormal_logpmf_with_location(shape, nuisance)
    if model == "gamma":
        return gamma_logpmf_with_scale(shape, np.exp(nuisance))
    if model == "negative_binomial":
        return negative_binomial_logpmf_with_mean(shape, np.exp(nuisance))
    raise KeyError(model)


def nuisance_initial(model: str, shape: float, hist: np.ndarray) -> np.ndarray:
    mu = np.maximum(means_from_hist(hist), 1e-9)
    if model == "power_cost":
        return power_logbeta(shape, hist)
    if model == "lognormal":
        return np.log(mu) - 0.5 * shape * shape
    if model == "gamma":
        return np.log(np.maximum(mu / shape, 1e-12))
    if model == "negative_binomial":
        return np.log(mu)
    raise KeyError(model)


def profile_nuisance(
    model: str,
    shape: float,
    hist: np.ndarray,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Likelihood profile with unbounded transformed nuisance parameters."""
    if model == "power_cost":
        return power_logbeta(shape, hist)
    z = nuisance_initial(model, shape, hist) if initial is None else np.asarray(initial, float).copy()
    n = np.maximum(hist.sum(axis=1), 1.0)

    def nll(values: np.ndarray) -> np.ndarray:
        logpmf = model_logpmf_with_nuisance(model, shape, values)
        return -np.sum(hist * logpmf, axis=1) / n

    step_size = 2e-4
    for _ in range(40):
        center = nll(z)
        plus = nll(z + step_size)
        minus = nll(z - step_size)
        grad = (plus - minus) / (2.0 * step_size)
        curvature = (plus - 2.0 * center + minus) / (step_size * step_size)
        if np.max(np.abs(grad)) < 2e-6:
            record_nuisance_audit(model, z, np.abs(grad))
            return z
        direction = -np.divide(grad, curvature, out=np.sign(grad).copy(), where=curvature > 1e-8)
        direction = np.clip(direction, -2, 2)
        direction[np.abs(grad) < 2e-6] = 0
        for _ in range(24):
            proposed = z + direction
            value = nll(proposed)
            accept = np.isfinite(value) & (value <= center + 1e-4 * grad * direction + 1e-13)
            if np.all(accept):
                break
            direction[~accept] *= 0.5
        z = np.where(accept, proposed, z)
    raise RuntimeError(f"{model} nuisance profile did not converge: max gradient {np.max(np.abs(grad))}")


def record_nuisance_audit(model: str, nuisance: np.ndarray, residual: np.ndarray) -> None:
    if not np.all(np.isfinite(nuisance)) or not np.all(np.isfinite(residual)):
        raise ValueError(f"nonfinite {model} nuisance fit")
    record = NUISANCE_AUDIT.setdefault(model, {
        "profile_calls": 0, "min_transformed_nuisance": float(np.min(nuisance)),
        "max_transformed_nuisance": float(np.max(nuisance)), "max_stationarity_residual": 0.0,
        "fixed_nuisance_bounds": None,
        "residual_definition": "relative power-moment error" if model == "power_cost" else "absolute per-trial NLL gradient",
    })
    record["profile_calls"] += 1
    record["min_transformed_nuisance"] = min(record["min_transformed_nuisance"], float(np.min(nuisance)))
    record["max_transformed_nuisance"] = max(record["max_transformed_nuisance"], float(np.max(nuisance)))
    record["max_stationarity_residual"] = max(record["max_stationarity_residual"], float(np.max(residual)))


def model_logpmf(model: str, shape: float, hist: np.ndarray) -> np.ndarray:
    nuisance = profile_nuisance(model, shape, hist)
    return model_logpmf_with_nuisance(model, shape, nuisance)


def per_neuron_log_score(logpmf: np.ndarray, test_hist: np.ndarray) -> np.ndarray:
    n = np.maximum(test_hist.sum(axis=1), 1.0)
    return np.sum(test_hist * logpmf, axis=1) / n


def fit_shared_shape(model: str, training_hist: np.ndarray, diagnostics: dict | None = None) -> float:
    lo, hi = BOUNDS[model]
    log_lo, log_hi = np.log(lo), np.log(hi)
    cache = {}

    def objective(log_shape: float) -> float:
        key = float(log_shape)
        if key in cache:
            return cache[key][0]
        shape = float(np.exp(log_shape))
        initial = None
        if cache:
            nearest = min(cache, key=lambda old: abs(old - key))
            if abs(nearest - key) < 0.3:
                initial = cache[nearest][1]
        nuisance = profile_nuisance(model, shape, training_hist, initial)
        logpmf = model_logpmf_with_nuisance(model, shape, nuisance)
        score = -float(np.mean(per_neuron_log_score(logpmf, training_hist)))
        if not np.isfinite(score):
            raise ValueError(f"nonfinite {model} shape objective")
        cache[key] = (score, nuisance)
        return score

    # Initial windows are numerical brackets, not declared parameter limits.
    # Expand either endpoint if it remains competitive with the fitted optimum.
    for expansion in range(9):
        result = minimize_scalar(
            objective,
            bounds=(log_lo, log_hi),
            method="bounded",
            options={"xatol": 5e-5, "maxiter": 80},
        )
        if not result.success:
            raise RuntimeError(f"{model} shared-shape optimizer failed: {result.message}")
        low_score, high_score = objective(log_lo), objective(log_hi)
        near_low = result.x - log_lo < 0.01 or low_score <= result.fun + 1e-8
        near_high = log_hi - result.x < 0.01 or high_score <= result.fun + 1e-8
        if not (near_low or near_high):
            break
        if near_low:
            log_lo -= np.log(2)
        if near_high:
            log_hi += np.log(2)
    else:
        raise RuntimeError(f"{model} shape optimum still touches an expanding search boundary; no result accepted")
    shape = float(np.exp(result.x))
    if diagnostics is not None:
        diagnostics.update({
            "initial_search_window": [lo, hi],
            "final_search_window": [float(np.exp(log_lo)), float(np.exp(log_hi))],
            "expansions": expansion, "shape": shape, "active_search_boundary": False,
            "n_unique_profile_objectives": len(cache), "training_nll_per_trial": float(result.fun),
            "lower_endpoint_nll_gap": float(low_score - result.fun),
            "upper_endpoint_nll_gap": float(high_score - result.fun),
            "profile_log_shape_gradient": float((objective(result.x + 1e-4) - objective(result.x - 1e-4)) / 2e-4),
        })
    return shape


def strict_qc(x: np.ndarray) -> bool:
    x = np.asarray(x, float)
    return bool(
        x.size >= 40
        and np.all(np.isfinite(x))
        and x.mean() >= MAIN.MIN_MEAN_COUNT
        and np.mean(x == 0) <= MAIN.MAX_ZERO_FRAC
    )


@lru_cache(maxsize=2)
def load_condition(condition: str) -> dict[str, list[np.ndarray]]:
    sessions = {}
    for session_file in MAIN.SESSIONS:
        source = MAIN.load_v1_counts(MAIN.FIG2 / session_file, condition)
        indices = [i for i, x in enumerate(source) if strict_qc(x)]
        name = session_file.replace(".mat", "")
        sessions[name] = [np.asarray(source[i], int) for i in indices]
        NEURON_IDS[(condition, name)] = np.asarray(indices) + 1
    return sessions


def split_neurons(rows: list[np.ndarray], rng: np.random.Generator):
    calibration, test = [], []
    for row in rows:
        idx = rng.permutation(row.size)
        cut = row.size // 2
        calibration.append(row[idx[:cut]])
        test.append(row[idx[cut:]])
    return hist_rows(calibration), hist_rows(test)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000):
    values = np.asarray(values, float)
    draws = np.empty(n_boot, float)
    for b in range(n_boot):
        draws[b] = np.mean(rng.choice(values, size=values.size, replace=True))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def evaluate_condition(condition: str, n_splits: int, rng: np.random.Generator, payload: dict) -> dict:
    sessions = load_condition(condition)
    folds = []
    for heldout, heldout_rows in sessions.items():
        training_rows = [row for name, rows in sessions.items() if name != heldout for row in rows]
        training_hist = hist_rows(training_rows)
        shape_audit = {model: {} for model in MODELS}
        shared_shapes = {model: fit_shared_shape(model, training_hist, shape_audit[model]) for model in MODELS}
        print(condition, heldout, "shapes", shared_shapes)

        split_scores = []
        split_calibration = []
        neuron_contrasts = {alt: [] for alt in MODELS if alt != "power_cost"}
        score_arrays = {model: [] for model in MODELS}
        nuisance_arrays = {model: [] for model in MODELS}
        for _ in range(n_splits):
            calibration_hist, test_hist = split_neurons(heldout_rows, rng)
            scores, calibration = {}, {}
            observed_zero = test_hist[:, 0] / np.maximum(test_hist.sum(axis=1), 1.0)
            observed_mean = means_from_hist(test_hist)
            for model in MODELS:
                nuisance = profile_nuisance(model, shared_shapes[model], calibration_hist)
                logpmf = model_logpmf_with_nuisance(model, shared_shapes[model], nuisance)
                scores[model] = per_neuron_log_score(logpmf, test_hist)
                score_arrays[model].append(scores[model])
                nuisance_arrays[model].append(nuisance)
                pmf = np.exp(logpmf)
                predicted_zero = pmf[:, 0]
                predicted_mean = np.sum(pmf * GRID[None, :], axis=1)
                calibration[model] = {
                    "zero_fraction_mae": float(np.mean(np.abs(predicted_zero - observed_zero))),
                    "mean_count_nrmse": float(
                        np.sqrt(np.mean((predicted_mean - observed_mean) ** 2))
                        / max(float(np.mean(observed_mean)), 1e-9)
                    ),
                }
            split_scores.append({model: float(np.mean(scores[model])) for model in MODELS})
            split_calibration.append(calibration)
            for alt in neuron_contrasts:
                neuron_contrasts[alt].append(scores["power_cost"] - scores[alt])

        mean_scores = {
            model: float(np.mean([split[model] for split in split_scores]))
            for model in MODELS
        }
        mean_calibration = {
            model: {
                metric: float(np.mean([split[model][metric] for split in split_calibration]))
                for metric in ("zero_fraction_mae", "mean_count_nrmse")
            }
            for model in MODELS
        }
        contrasts = {}
        prefix = f"{condition}_{heldout}"
        payload[f"{prefix}_neuron_index_1based"] = NEURON_IDS[(condition, heldout)]
        for model in MODELS:
            payload[f"{prefix}_{model}"] = np.mean(score_arrays[model], axis=0)
            payload[f"{prefix}_split_{model}"] = np.vstack(score_arrays[model])
            payload[f"{prefix}_nuisance_{model}"] = np.vstack(nuisance_arrays[model])
        for alt, arrays in neuron_contrasts.items():
            per_neuron = np.mean(np.vstack(arrays), axis=0)
            contrasts[alt] = {
                "mean_delta_elpd_per_trial": float(np.mean(per_neuron)),
                "median_delta_elpd_per_trial": float(np.median(per_neuron)),
                "neuron_bootstrap_ci95_of_mean": bootstrap_ci(per_neuron, rng),
                "fraction_neurons_power_cost_better": float(np.mean(per_neuron > 0)),
            }
        folds.append({
            "heldout_session": heldout,
            "n_neurons": len(heldout_rows),
            "shared_shapes_trained_on_other_sessions": shared_shapes,
            "shape_bound_audit": shape_audit,
            "calibration_nuisance_ranges": {
                model: [float(np.min(nuisance_arrays[model])), float(np.max(nuisance_arrays[model]))]
                for model in MODELS
            },
            "mean_elpd_per_trial": mean_scores,
            "best_model": max(mean_scores, key=mean_scores.get),
            "calibration": mean_calibration,
            "power_cost_contrasts": contrasts,
        })

    winners = {model: int(sum(fold["best_model"] == model for fold in folds)) for model in MODELS}
    session_mean = {model: float(np.mean([fold["mean_elpd_per_trial"][model] for fold in folds])) for model in MODELS}
    contrasts = {}
    for alt in MODELS:
        if alt == "power_cost":
            continue
        values = np.asarray([
            fold["power_cost_contrasts"][alt]["mean_delta_elpd_per_trial"] for fold in folds
        ])
        grouped = {}
        for prefix in sorted({fold["heldout_session"][:3] for fold in folds}):
            grouped[prefix] = float(np.mean([
                fold["power_cost_contrasts"][alt]["mean_delta_elpd_per_trial"]
                for fold in folds if fold["heldout_session"].startswith(prefix)
            ]))
        contrasts[alt] = {
            "per_session_mean_delta_elpd_per_trial": values.tolist(),
            "sessions_power_cost_better": int(np.sum(values > 0)),
            "mean_across_sessions": float(np.mean(values)),
            "session_code_group_means": grouped,
            "session_code_groups_power_cost_better": int(np.sum(np.asarray(list(grouped.values())) > 0)),
        }
    return {
        "folds": folds,
        "session_winners": winners,
        "mean_across_heldout_sessions": session_mean,
        "power_cost_contrast_summary": contrasts,
    }


def sample_from_pmf(logpmf: np.ndarray, n_trials: int, rng: np.random.Generator) -> list[np.ndarray]:
    rows = []
    for row in np.exp(logpmf):
        rows.append(rng.choice(GRID.astype(int), size=n_trials, replace=True, p=row / row.sum()))
    return rows


def model_recovery(condition: str, rng: np.random.Generator, payload: dict, n_neurons: int = 60) -> dict:
    sessions = load_condition(condition)
    rows = [row for values in sessions.values() for row in values]
    hist = hist_rows(rows)
    shape_audit = {model: {} for model in MODELS}
    shapes = {model: fit_shared_shape(model, hist, shape_audit[model]) for model in MODELS}
    means = means_from_hist(hist)
    selected_means = rng.choice(means, size=min(n_neurons, means.size), replace=False)
    mean_hist = np.zeros((selected_means.size, COUNT_MAX + 1), float)
    nearest = np.clip(np.rint(selected_means).astype(int), 0, COUNT_MAX)
    mean_hist[np.arange(selected_means.size), nearest] = 1.0

    matrix = {}
    for generator in MODELS:
        generating_logpmf = model_logpmf(generator, shapes[generator], mean_hist)
        simulated = sample_from_pmf(generating_logpmf, 400, rng)
        calibration_hist, test_hist = split_neurons(simulated, rng)
        score_matrix = np.column_stack([
            per_neuron_log_score(model_logpmf(model, shapes[model], calibration_hist), test_hist)
            for model in MODELS
        ])
        winners = np.argmax(score_matrix, axis=1)
        payload[f"recovery_{condition}_{generator}_scores"] = score_matrix
        payload[f"recovery_{condition}_{generator}_winners"] = winners
        matrix[generator] = {
            model: float(np.mean(winners == i)) for i, model in enumerate(MODELS)
        }
    return {
        "fitted_shapes_all_sessions": shapes, "winner_fraction": matrix,
        "shape_bound_audit": shape_audit, "n_synthetic_neurons_per_generator": len(selected_means),
        "n_counts_per_synthetic_neuron": 400,
        "scope": "conditional fixed-shape separability; the generating family is scored with its generating shape, not a recovery of LOSO shape learning",
    }


def make_figure(output: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.0), constrained_layout=True)
    colors = {"lognormal": "#9b59b6", "gamma": "#d9822b", "negative_binomial": "#4e79a7"}
    for ax, condition, title in zip(axes[:2], ("spontaneous", "driven"), ("A  Spontaneous", "B  Driven")):
        folds = output["conditions"][condition]["folds"]
        x = np.arange(len(folds))
        for alt, offset in zip(("lognormal", "gamma", "negative_binomial"), (-0.18, 0, 0.18)):
            values = [fold["power_cost_contrasts"][alt]["mean_delta_elpd_per_trial"] for fold in folds]
            ax.scatter(x + offset, values, s=38, color=colors[alt], label=DISPLAY[alt], zorder=3)
            ax.plot(x + offset, values, color=colors[alt], alpha=0.35, lw=1)
        ax.axhline(0, color="0.25", lw=1)
        ax.set_xticks(x, [fold["heldout_session"].replace("l", "-").replace("r", "-")[:7] for fold in folds])
        ax.set_xlabel("held-out recording session")
        ax.set_ylabel(r"$\Delta$ELPD/trial: power-cost $-$ comparator")
        ax.set_title(title)
        if condition == "spontaneous":
            ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    recovery = output["model_recovery"]["spontaneous"]["winner_fraction"]
    mat = np.asarray([[recovery[g][m] for m in MODELS] for g in MODELS])
    image = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] > 0.55 else "black", fontsize=8)
    ax.set_xticks(range(len(MODELS)), [DISPLAY[m] for m in MODELS], rotation=35, ha="right")
    ax.set_yticks(range(len(MODELS)), [DISPLAY[m] for m in MODELS])
    ax.set_xlabel("selected model")
    ax.set_ylabel("generating model")
    ax.set_title("C  Conditional fixed-shape recovery")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="winner fraction")
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(n_splits: int) -> None:
    started = time.perf_counter()
    if n_splits < 1:
        raise ValueError("at least one split is required")
    NUISANCE_AUDIT.clear()
    rng = np.random.default_rng(SEED)
    payload = {}
    output = {
        "analysis_version": "20260907_adaptive_shape_unbounded_log_nuisance",
        "seed": SEED,
        "n_trial_splits_within_heldout_session": n_splits,
        "independent_unit": "held-out recording session (five sessions documented from three animals)",
        "animal_grouping_note": (
            "The 105/106/107 session-code prefixes are reported as descriptive three-animal groups; "
            "the five held-out sessions remain the unit used for predictive comparisons."
        ),
        "parameterization": (
            "one shape parameter shared across four training sessions plus one neuron-specific "
            "rate/scale parameter profiled by native-count likelihood on the held-out neuron's "
            "calibration trials"
        ),
        "nuisance_calibration_note": (
            "At fixed shared shape, each family receives exactly one neuron-level nuisance parameter. "
            "That parameter is profiled against the native-count calibration likelihood for every "
            "family; scoring is performed only on the disjoint half."
        ),
        "parameter_domain": {
            "count_support": [0, COUNT_MAX],
            "power_cost": "alpha>0, beta>0; unbounded log-beta; positive-cost active branch only",
            "lognormal": "sigma>0, real location",
            "gamma": "shape>0, scale>0; unbounded log-scale",
            "negative_binomial": "size>0, mean>0; unbounded log-mean",
            "shape_search": "positive shapes; initial windows expand if an endpoint is competitive; unresolved boundary fits raise rather than being accepted",
            "probability_floor": PROB_FLOOR,
        },
        "qc_note": "Strict QC is determined using all trials before calibration/test splitting; predictions are conditional on retrospectively QC-selected neurons",
        "training_objective": "mean per-neuron per-trial native-count NLL; nuisance is reprofiled at every shared-shape objective; objective evaluations cached within each fit",
        "models": {model: DISPLAY[model] for model in MODELS},
        "conditions": {},
        "model_recovery": {},
    }
    for condition in ("spontaneous", "driven"):
        condition_started = time.perf_counter()
        output["conditions"][condition] = evaluate_condition(condition, n_splits, rng, payload)
        output["model_recovery"][condition] = model_recovery(condition, rng, payload)
        print(condition, "complete in", round(time.perf_counter() - condition_started, 1), "seconds", flush=True)
    output["nuisance_bound_and_convergence_audit"] = NUISANCE_AUDIT
    if not all(np.all(np.isfinite(value)) for value in payload.values()):
        raise ValueError("nonfinite native-count output arrays")
    np.savez_compressed(OUT_NPZ, **payload)
    output["provenance"] = {
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loader_code_sha256": hashlib.sha256(MAIN_SCRIPT.read_bytes()).hexdigest(),
        "input_sha256": {name: hashlib.sha256((MAIN.FIG2 / name).read_bytes()).hexdigest() for name in MAIN.SESSIONS},
        "npz_sha256": hashlib.sha256(OUT_NPZ.read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "continuous_pipeline_rerun": False, "entropy_recomputed": False,
    }
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    OUT_JSON.write_text(serialized, encoding="utf-8")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / OUT_JSON.name).write_text(serialized, encoding="utf-8")
    (REVIEW_DIR / OUT_NPZ.name).write_bytes(OUT_NPZ.read_bytes())
    input_payload = {}
    for condition in ("spontaneous", "driven"):
        for name, rows in load_condition(condition).items():
            input_payload[f"{condition}_{name}_counts"] = np.asarray(rows, np.int16)
            input_payload[f"{condition}_{name}_neuron_index_1based"] = NEURON_IDS[(condition, name)]
    np.savez_compressed(REVIEW_DIR / "native_count_input_cache.npz", **input_payload)
    make_figure(output)
    print("Wrote", OUT_JSON, OUT_FIG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    args = parser.parse_args()
    main(args.splits)
