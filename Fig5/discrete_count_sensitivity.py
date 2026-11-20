"""Held-out discrete-count sensitivity analysis for the V1 response-law test.

The main Fig. 6 analysis continuity-jitters spike counts before fitting positive
continuous densities.  This script keeps the observations discrete and compares
four two-parameter count models on their native likelihood:

* bounded power-cost MaxEnt: p(k) propto exp(-beta * k**alpha)
* discretized lognormal
* discretized gamma
* negative binomial

All models use the physical 1.28-s count support k=0,...,1280.  Results are
reported by recording session and under both the main-text QC and a broader QC
that retains low-rate/zero-heavy neurons.  No manuscript figure is written.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, nbinom, wilcoxon


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN_SCRIPT = HERE / "discriminative_maxent_test.py"
OUT_JSON = HERE / "discrete_count_sensitivity.json"

SEED = 20260629
COUNT_MAX = 1280
ALPHA_BOUNDS = (0.20, 8.0)
PROB_FLOOR = 1e-300


def load_main_module():
    spec = importlib.util.spec_from_file_location("discriminative_main", MAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN = load_main_module()
GRID = np.arange(COUNT_MAX + 1, dtype=float)


def _hist(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, int)
    if np.any((x < 0) | (x > COUNT_MAX)):
        raise ValueError("count outside the declared 0..1280 support")
    return np.bincount(x, minlength=COUNT_MAX + 1).astype(float)


def _weighted_nll(logp: np.ndarray, counts: np.ndarray) -> float:
    if logp.shape != counts.shape or not np.all(np.isfinite(logp)):
        return 1e100
    return -float(np.dot(counts, logp))


def power_logpmf(alpha: float, beta: float) -> np.ndarray:
    logw = -beta * np.power(GRID, alpha)
    return logw - logsumexp(logw)


def fit_power(train: np.ndarray) -> tuple[float, float]:
    counts = _hist(train)
    mu = max(float(np.mean(train)), 1e-6)
    cv = float(np.std(train, ddof=1) / mu) if train.size > 1 else 1.0
    alpha0 = MAIN.alpha_from_cv(cv)
    if not np.isfinite(alpha0):
        alpha0 = 1.0
    alpha0 = float(np.clip(alpha0, *ALPHA_BOUNDS))
    moment = max(float(np.mean(np.power(train, alpha0))), 1e-6)
    beta0 = float(np.clip(1.0 / (alpha0 * moment), 1e-6, 100.0))

    def objective(z):
        alpha, beta = np.exp(z)
        return _weighted_nll(power_logpmf(alpha, beta), counts)

    fit = minimize(
        objective,
        np.log([alpha0, beta0]),
        method="L-BFGS-B",
        bounds=[(np.log(ALPHA_BOUNDS[0]), np.log(ALPHA_BOUNDS[1])),
                (np.log(1e-6), np.log(100.0))],
    )
    alpha, beta = np.exp(fit.x)
    return float(alpha), float(beta)


def _discretized_logpdf(dist, upper_cdf: float) -> np.ndarray:
    lower = np.maximum(GRID - 0.5, 0.0)
    upper = GRID + 0.5
    probs = dist.cdf(upper) - dist.cdf(lower)
    probs = np.clip(probs / max(upper_cdf, PROB_FLOOR), PROB_FLOOR, 1.0)
    return np.log(probs)


def lognormal_logpmf(m: float, sigma: float) -> np.ndarray:
    dist = lognorm(s=sigma, scale=np.exp(m), loc=0.0)
    return _discretized_logpdf(dist, float(dist.cdf(COUNT_MAX + 0.5)))


def fit_lognormal(train: np.ndarray) -> tuple[float, float]:
    counts = _hist(train)
    lx = np.log(np.asarray(train, float) + 0.5)
    x0 = np.array([float(lx.mean()), np.log(float(lx.std(ddof=1) + 1e-3))])

    def objective(z):
        return _weighted_nll(lognormal_logpmf(float(z[0]), float(np.exp(z[1]))), counts)

    fit = minimize(
        objective, x0, method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (np.log(1e-3), np.log(10.0))],
    )
    return float(fit.x[0]), float(np.exp(fit.x[1]))


def gamma_logpmf(shape: float, scale: float) -> np.ndarray:
    dist = gamma_dist(a=shape, scale=scale, loc=0.0)
    return _discretized_logpdf(dist, float(dist.cdf(COUNT_MAX + 0.5)))


def fit_gamma(train: np.ndarray) -> tuple[float, float]:
    counts = _hist(train)
    x = np.asarray(train, float) + 0.5
    mu = max(float(x.mean()), 1e-6)
    var = max(float(x.var(ddof=1)), 1e-6)
    shape0 = float(np.clip(mu * mu / var, 1e-3, 1e3))
    scale0 = float(np.clip(var / mu, 1e-3, 1e3))

    def objective(z):
        shape, scale = np.exp(z)
        return _weighted_nll(gamma_logpmf(shape, scale), counts)

    fit = minimize(
        objective, np.log([shape0, scale0]), method="L-BFGS-B",
        bounds=[(np.log(1e-3), np.log(1e3)),
                (np.log(1e-3), np.log(1e3))],
    )
    shape, scale = np.exp(fit.x)
    return float(shape), float(scale)


def nb_logpmf(size: float, prob: float) -> np.ndarray:
    logp = nbinom.logpmf(GRID, size, prob)
    norm = float(nbinom.logcdf(COUNT_MAX, size, prob))
    return logp - norm


def fit_negative_binomial(train: np.ndarray) -> tuple[float, float]:
    counts = _hist(train)
    x = np.asarray(train, float)
    mu = max(float(x.mean()), 1e-6)
    var = max(float(x.var(ddof=1)), mu + 1e-6)
    size0 = float(np.clip(mu * mu / max(var - mu, 1e-6), 1e-3, 1e4))
    prob0 = float(np.clip(size0 / (size0 + mu), 1e-5, 1 - 1e-5))

    def objective(z):
        size = float(np.exp(z[0]))
        prob = float(np.clip(expit(z[1]), 1e-8, 1 - 1e-8))
        return _weighted_nll(nb_logpmf(size, prob), counts)

    fit = minimize(
        objective,
        [np.log(size0), np.log(prob0 / (1.0 - prob0))],
        method="L-BFGS-B",
        bounds=[(np.log(1e-3), np.log(1e5)), (-18.0, 18.0)],
    )
    return float(np.exp(fit.x[0])), float(expit(fit.x[1]))


def held_out_compare(counts: np.ndarray, rng: np.random.Generator, n_splits: int) -> dict:
    x = np.asarray(counts, int)
    contrasts = {"lognormal": [], "gamma": [], "negative_binomial": []}
    alphas = []
    for _ in range(n_splits):
        idx = rng.permutation(x.size)
        n_train = x.size // 2
        train, test = x[idx[:n_train]], x[idx[n_train:]]
        test_hist = _hist(test)

        alpha, beta = fit_power(train)
        m, sigma = fit_lognormal(train)
        shape, scale = fit_gamma(train)
        size, prob = fit_negative_binomial(train)

        ll_power = -_weighted_nll(power_logpmf(alpha, beta), test_hist) / test.size
        alternatives = {
            "lognormal": -_weighted_nll(lognormal_logpmf(m, sigma), test_hist) / test.size,
            "gamma": -_weighted_nll(gamma_logpmf(shape, scale), test_hist) / test.size,
            "negative_binomial": -_weighted_nll(nb_logpmf(size, prob), test_hist) / test.size,
        }
        for name, ll_alt in alternatives.items():
            contrasts[name].append(float(ll_power - ll_alt))
        alphas.append(alpha)

    return {
        "dloglik_power_minus_lognormal": float(np.mean(contrasts["lognormal"])),
        "dloglik_power_minus_gamma": float(np.mean(contrasts["gamma"])),
        "dloglik_power_minus_negative_binomial": float(np.mean(contrasts["negative_binomial"])),
        "effective_shape_mle": float(np.mean(alphas)),
        "mean_count": float(x.mean()),
        "zero_fraction": float(np.mean(x == 0)),
    }


def passes_qc(x: np.ndarray, mode: str) -> bool:
    x = np.asarray(x, float)
    if x.size < 40 or not np.all(np.isfinite(x)):
        return False
    if mode == "strict":
        return bool(x.mean() >= MAIN.MIN_MEAN_COUNT and np.mean(x == 0) <= MAIN.MAX_ZERO_FRAC)
    if mode == "broad":
        return bool(x.mean() >= 0.1 and np.mean(x == 0) < 0.99)
    raise ValueError(mode)


def summarize(results: list[dict]) -> dict:
    out = {"n_neurons": len(results)}
    for alt in ("lognormal", "gamma", "negative_binomial"):
        vals = np.asarray([r[f"dloglik_power_minus_{alt}"] for r in results], float)
        out[f"power_win_rate_vs_{alt}"] = float(np.mean(vals > 0))
        out[f"median_dloglik_power_minus_{alt}"] = float(np.median(vals))
        out[f"mean_dloglik_power_minus_{alt}"] = float(np.mean(vals))
    out["effective_shape_median"] = float(np.median([r["effective_shape_mle"] for r in results]))
    return out


def summarize_sessions(results: list[dict]) -> dict:
    rows = []
    for session in sorted({r["session"] for r in results}):
        sub = [r for r in results if r["session"] == session]
        row = {"session": session, **summarize(sub)}
        rows.append(row)
    inference = {}
    for alt in ("lognormal", "gamma", "negative_binomial"):
        medians = np.asarray([r[f"median_dloglik_power_minus_{alt}"] for r in rows])
        inference[alt] = {
            "sessions_with_positive_median": int(np.sum(medians > 0)),
            "n_sessions": int(medians.size),
            "session_median_of_medians": float(np.median(medians)),
            "session_wilcoxon_p_two_sided": (
                float(wilcoxon(medians).pvalue) if medians.size and np.any(medians != 0) else 1.0
            ),
        }
    return {"by_session": rows, "session_level_inference": inference}


def run_condition(condition: str, qc_mode: str, n_splits: int, rng) -> dict:
    results = []
    for session_file in MAIN.SESSIONS:
        path = MAIN.FIG2 / session_file
        if not path.exists():
            continue
        for neuron_index, counts in enumerate(MAIN.load_v1_counts(path, condition)):
            if not passes_qc(counts, qc_mode):
                continue
            rec = held_out_compare(counts, rng, n_splits)
            rec.update({
                "session": session_file.replace(".mat", ""),
                "neuron_index": int(neuron_index),
            })
            results.append(rec)
    return {"pooled": summarize(results), **summarize_sessions(results)}


def main(n_splits: int, qc_modes: list[str]) -> None:
    output = {
        "seed": SEED,
        "n_splits": n_splits,
        "count_support": [0, COUNT_MAX],
        "models": ["bounded power-cost MaxEnt", "discretized lognormal", "discretized gamma", "negative binomial"],
        "note": "All candidates have two fitted parameters and are scored on held-out integer counts without continuity jitter.",
    }
    for qc_mode in qc_modes:
        rng = np.random.default_rng(SEED + (0 if qc_mode == "strict" else 10000))
        output[qc_mode] = {}
        for condition in ("driven", "spontaneous"):
            print(f"[{qc_mode}] {condition}")
            output[qc_mode][condition] = run_condition(condition, qc_mode, n_splits, rng)
            print(json.dumps(output[qc_mode][condition]["pooled"], indent=2))
    OUT_JSON.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--qc", choices=["strict", "broad", "both"], default="both")
    args = parser.parse_args()
    modes = ["strict", "broad"] if args.qc == "both" else [args.qc]
    main(args.splits, modes)
