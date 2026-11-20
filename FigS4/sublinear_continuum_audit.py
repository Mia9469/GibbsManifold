#!/usr/bin/env python3
"""Correct S12 numerics without modifying the legacy sweep or manuscript.

Run from any directory:
    python3 FigS4/sublinear_continuum_audit.py

For t=beta*A, z=t*R**alpha, s=(j+1)/alpha,
    integral(r**j * exp(-t*r**alpha), 0, R)
        = gamma_lower(s, z) / (alpha * t**s).
The differential entropy uses fixed numerical response units: h=beta*C+logZ.
The figure connects the original 80 beta nodes in beta order, not CV order.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.io import loadmat
from scipy.optimize import brentq
from scipy.special import gammainc, gammaln


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/final_text_review_20260907/theory_corrected"
ALPHAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BETAS = np.logspace(-4, np.log10(5.0), 80)
A_COST = 0.37
R_MAX = 100.0
MARKER_CVS = {"evoked": 0.63, "spontaneous": 0.86}


def log_integral(power, alpha, beta, cost_scale=A_COST, r_max=R_MAX):
    """Log of the unnormalized power moment, using lower incomplete gamma."""
    values = (power, alpha, beta, cost_scale, r_max)
    if not all(np.isfinite(x) for x in values):
        raise ValueError("Parameters must be finite")
    if power < 0 or alpha <= 0 or beta < 0 or cost_scale <= 0 or r_max <= 0:
        raise ValueError("Require power,beta >= 0 and alpha,A,R > 0")
    if beta == 0:
        return (power + 1) * np.log(r_max) - np.log(power + 1)
    t = beta * cost_scale
    z = t * r_max**alpha
    shape = (power + 1) / alpha
    regularized = gammainc(shape, z)
    if regularized <= 0:
        raise FloatingPointError("Incomplete gamma underflow outside tested range")
    return gammaln(shape) + np.log(regularized) - np.log(alpha) - shape * np.log(t)


def state(alpha, beta, cost_scale=A_COST, r_max=R_MAX):
    """Continuum moments, entropy yield, and analytic beta derivatives."""
    log_z = log_integral(0, alpha, beta, cost_scale, r_max)

    def moment(power):
        return np.exp(log_integral(power, alpha, beta, cost_scale, r_max) - log_z)

    mean = moment(1)
    second = moment(2)
    variance = second - mean**2
    cost = cost_scale * moment(alpha)
    cost_variance = cost_scale**2 * moment(2 * alpha) - cost**2
    if variance <= 0 or cost_variance <= 0 or cost <= 0:
        raise ArithmeticError("Nonpositive variance or cost in a nondegenerate law")
    cv_squared = variance / mean**2
    z = beta * cost_scale * r_max**alpha

    # Avoid roundoff above CV=1 when the linear branch saturates exponentially.
    if alpha == 1.0 and z >= 2:
        q = np.exp(-z)
        cv_squared = 1 - z * q * ((z - 2) + (z + 2) * q) / (1 - (z + 1) * q)**2
    cv = np.sqrt(cv_squared)
    entropy = beta * cost + log_z

    if beta == 0:
        cov_rc = cost_scale * moment(1 + alpha) - mean * cost
        cov_deviation = cost_scale * (
            moment(2 + alpha) - 2 * mean * moment(1 + alpha) + mean**2 * moment(alpha)
        ) - variance * cost
        cv_derivative = cv * (cov_rc / mean - cov_deviation / (2 * variance))
    else:
        # Differentiate gamma(s0,z)*gamma(s2,z)/gamma(s1,z)^2.
        # Boundary terms remain resolvable when ordinary moment subtraction
        # loses the exponentially small positive derivative near saturation.
        boundary = []
        for j in (0, 1, 2):
            shape = (j + 1) / alpha
            log_gamma = gammaln(shape) + np.log(gammainc(shape, z))
            boundary.append(np.exp((shape - 1) * np.log(z) - z - log_gamma))
        derivative_log_ratio = cost_scale * r_max**alpha * (
            boundary[0] + boundary[2] - 2 * boundary[1]
        )
        cv_derivative = (1 + cv_squared) * derivative_log_ratio / (2 * cv)

    return {key: float(value) for key, value in {
        "beta": beta, "mean": mean, "second_moment": second,
        "variance": variance, "C": cost, "variance_cost": cost_variance,
        "CV": cv, "logZ": log_z, "h": entropy,
        "entropy_yield": entropy / cost, "dCV_dbeta": cv_derivative,
        "dyield_dbeta": cost_variance * log_z / cost**2,
    }.items()}


def density(r, alpha, beta):
    r = np.asarray(r, dtype=float)
    return np.where(
        (r >= 0) & (r <= R_MAX),
        np.exp(-beta * A_COST * np.maximum(r, 0)**alpha - log_integral(0, alpha, beta)),
        0.0,
    )


def yield_turn(alpha):
    """Numerical root of the exact condition logZ=0, including an outside flag."""
    upper = float(BETAS[-1])
    while log_integral(0, alpha, upper) > 0:
        upper *= 2
    beta = brentq(lambda b: log_integral(0, alpha, b), 0.0, upper, xtol=1e-13)
    result = state(alpha, beta)
    result["inside_original_beta_sweep"] = bool(BETAS[0] <= beta <= BETAS[-1])
    return result


def marker_state(cv, alpha=0.7):
    beta = brentq(
        lambda b: state(alpha, b)["CV"] - cv,
        float(BETAS[0]), float(BETAS[-1]), xtol=1e-13,
    )
    return state(alpha, beta)


def quadrature_state(alpha, beta):
    """Independent adaptive integration in r, not incomplete-gamma space."""
    def integral(power):
        result, error = quad(
            lambda r: r**power * np.exp(-beta * A_COST * r**alpha),
            0, R_MAX, epsabs=1e-11, epsrel=2e-11, limit=400,
        )
        return result, error

    z, z_error = integral(0)
    mean = integral(1)[0] / z
    second = integral(2)[0] / z
    cost = A_COST * integral(alpha)[0] / z
    variance = second - mean**2
    log_z = np.log(z)
    entropy_direct = quad(
        lambda r: -(np.exp(-beta * A_COST * r**alpha) / z)
        * (-beta * A_COST * r**alpha - log_z),
        0, R_MAX, epsabs=1e-11, epsrel=2e-11, limit=400,
    )[0]
    cost_variance = A_COST**2 * integral(2 * alpha)[0] / z - cost**2
    cov_rc = A_COST * integral(1 + alpha)[0] / z - mean * cost
    cov_deviation = quad(
        lambda r: (r - mean)**2 * (A_COST * r**alpha - cost)
        * np.exp(-beta * A_COST * r**alpha) / z,
        0, R_MAX, epsabs=1e-11, epsrel=2e-11, limit=400,
    )[0]
    cv = np.sqrt(variance) / mean
    return {
        "mean": mean, "second_moment": second, "variance": variance,
        "C": cost, "variance_cost": cost_variance, "CV": cv,
        "logZ": log_z, "h": entropy_direct, "entropy_yield": entropy_direct / cost,
        "dCV_dbeta": cv * (cov_rc / mean - cov_deviation / (2 * variance)),
        "normalizer_quad_error": z_error,
    }


def validate():
    cases = []
    keys = ("mean", "second_moment", "variance", "C", "variance_cost", "CV", "logZ", "h", "entropy_yield")
    for alpha in ALPHAS:
        for beta in (1e-4, 0.3, 5.0):
            exact = state(alpha, beta)
            reference = quadrature_state(alpha, beta)
            errors = {key: abs(exact[key] - reference[key]) / max(1, abs(reference[key])) for key in keys}
            if max(errors.values()) > 2e-9:
                raise AssertionError((alpha, beta, errors))
            cases.append({"alpha": alpha, "beta": beta, "scaled_absolute_errors": errors})

    derivative_errors = []
    for alpha in ALPHAS:
        beta = 0.3
        exact = state(alpha, beta)
        q = quadrature_state(alpha, beta)
        error = abs(exact["dCV_dbeta"] - q["dCV_dbeta"])
        if error > 2e-8:
            raise AssertionError((alpha, "CV derivative", error))
        eps = 1e-5
        plus, minus = state(alpha, beta + eps), state(alpha, beta - eps)
        finite_difference = (plus["entropy_yield"] - minus["entropy_yield"]) / (2 * eps)
        if abs(finite_difference - exact["dyield_dbeta"]) > 2e-7:
            raise AssertionError((alpha, "yield derivative"))
        derivative_errors.append(error)

    with mp.workdps(160):
        k = mp.mpf("5") * mp.mpf("0.37") * 100
        q = mp.exp(-k)
        deficit_squared = k * q * ((k - 2) + (k + 2) * q) / (1 - (k + 1) * q)**2
        cv = mp.sqrt(1 - deficit_squared)
        gap = mp.nstr(1 - cv, 30)
        if not 0 < cv < 1 or abs(state(1.0, 5.0)["CV"] - float(cv)) > 1e-13:
            raise AssertionError("Linear high-beta CV check failed")

    for alpha in ALPHAS:
        uniform = state(alpha, 0)
        np.testing.assert_allclose(
            [uniform["mean"], uniform["variance"], uniform["CV"], uniform["h"]],
            [R_MAX / 2, R_MAX**2 / 12, 1 / np.sqrt(3), np.log(R_MAX)], rtol=2e-12,
        )
        turn = yield_turn(alpha)
        assert abs(turn["logZ"]) < 1e-11
        assert state(alpha, turn["beta"] * 0.99)["dyield_dbeta"] > 0
        assert state(alpha, turn["beta"] * 1.01)["dyield_dbeta"] < 0

    return {
        "passed": True, "adaptive_quad_cases": cases,
        "quad_scaled_absolute_tolerance": 2e-9,
        "max_quad_scaled_absolute_error": max(max(x["scaled_absolute_errors"].values()) for x in cases),
        "max_CV_derivative_vs_quad_absolute_error": max(derivative_errors),
        "linear_alpha1_beta5_CV": state(1.0, 5.0)["CV"],
        "linear_alpha1_beta5_exact_gap_below_CV1": gap,
        "additional_checks": ["uniform endpoint", "yield derivatives", "root sign changes"],
    }


def make_records():
    records = []
    for alpha in ALPHAS:
        nodes = [state(alpha, float(beta)) for beta in BETAS]
        cvs = np.array([x["CV"] for x in nodes])
        lo = max(0, int(np.searchsorted(cvs, 0.63)) - 1)
        hi = min(len(nodes) - 1, int(np.searchsorted(cvs, 0.86)))
        selected = nodes[lo:hi + 1]
        yields = np.array([x["entropy_yield"] for x in nodes])
        downward = np.flatnonzero(np.diff(yields) < 0)
        records.append({
            "alpha": alpha, "nodes": nodes, "yield_turn_logZ_zero": yield_turn(alpha),
            "negative_yield_steps": len(downward),
            "negative_yield_step_start_indices_zero_based": downward.tolist(),
            "negative_CV_steps_beyond_1e-10_roundoff": int(np.sum(np.diff(cvs) < -1e-10)),
            "sampled_CV_derivatives_all_positive": all(x["dCV_dbeta"] > 0 for x in nodes),
            "marker_bracketing_nodes": {
                "start_index_zero_based": lo, "end_index_zero_based": hi,
                "beta_range": [nodes[lo]["beta"], nodes[hi]["beta"]],
                "CV_range": [nodes[lo]["CV"], nodes[hi]["CV"]],
                "min_logZ": min(x["logZ"] for x in selected),
                "min_dCV_dbeta": min(x["dCV_dbeta"] for x in selected),
                "min_dyield_dbeta": min(x["dyield_dbeta"] for x in selected),
                "positive_adjacent_yield_increments": bool(np.all(np.diff(yields[lo:hi + 1]) > 0)),
            },
        })
    return records


def legacy_comparison(records):
    old = loadmat(ROOT / "FigS4/alpha_sweep_results.mat", squeeze_me=True)
    np.testing.assert_allclose(old["alpha_list"], ALPHAS, atol=1e-14, rtol=0)
    np.testing.assert_allclose(old["beta_grid"], BETAS, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose([old["A_cost"], old["r_max"]], [A_COST, R_MAX])
    result = []
    for i, record in enumerate(records):
        new = record["nodes"][-1]
        result.append({
            "alpha": record["alpha"], "beta": new["beta"],
            "legacy_grid_CV": float(old["CV_arr"][i, -1]), "continuum_CV": new["CV"],
            "legacy_grid_h": float(old["H_arr"][i, -1]), "continuum_h": new["h"],
            "legacy_grid_yield": float(old["ICE_arr"][i, -1]), "continuum_yield": new["entropy_yield"],
        })
    return result


def legacy_hashes():
    paths = [ROOT / "FigS4/alpha_sweep_results.mat", ROOT / "FigS4/alpha_sweep_sublinear.m"]
    for folder in (ROOT / "FigS4", ROOT / "updated/figures"):
        for stem in ("FigureS9", "FigureS12", "FigureS13"):
            paths.extend(folder / (stem + ext) for ext in (".pdf", ".png"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def plot_figure(records, markers, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.22, top=0.81, wspace=0.35)
    colors = plt.get_cmap("viridis")(np.linspace(0.04, 0.90, len(ALPHAS)))
    r = R_MAX * np.linspace(0, 1, 1500)**2
    for record, color in zip(records, colors):
        alpha = record["alpha"]
        nodes = record["nodes"]
        axes[0].plot(r, density(r, alpha, 0.3), color=color, lw=1.8, label=f"{alpha:.1f}")
        axes[1].plot([x["CV"] for x in nodes], [x["entropy_yield"] for x in nodes], color=color, lw=1.9)
        turn = record["yield_turn_logZ_zero"]
        if turn["inside_original_beta_sweep"]:
            axes[1].plot(turn["CV"], turn["entropy_yield"], "D", ms=5, mfc="white", mec=color, mew=1.2)
        axes[1].plot(nodes[-1]["CV"], nodes[-1]["entropy_yield"], "s", ms=4, color=color)

    axes[0].set(title="A  Response densities", xlabel="Response r (fixed rate units)", ylabel=r"Probability density $p(r)$", xlim=(0, R_MAX))
    axes[0].text(0.97, 0.97, r"$\beta=0.3$", ha="right", va="top", transform=axes[0].transAxes)
    axes[0].legend(title=r"Cost exponent $\alpha$", ncol=2, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.84), fontsize=8, title_fontsize=8)
    axes[1].set(title="B  Entropy yield by branch", xlabel="Coefficient of variation", ylabel=r"Entropy yield $h/C$ (nats / cost unit)")
    axes[1].plot([], [], "D", ms=5, mfc="white", mec="#404040", label=r"Yield maximum: $\log Z=0$")
    axes[1].plot([], [], "s", ms=4, color="#404040", label=r"Sweep endpoint: $\beta=5$")
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    reference = records[ALPHAS.index(0.7)]
    axes[2].plot([x["CV"] for x in reference["nodes"]], [x["entropy_yield"] for x in reference["nodes"]], color="#303030", lw=2)
    turn, endpoint = reference["yield_turn_logZ_zero"], reference["nodes"][-1]
    axes[2].plot(turn["CV"], turn["entropy_yield"], "D", ms=5, mfc="white", mec="#303030")
    axes[2].plot(endpoint["CV"], endpoint["entropy_yield"], "s", ms=4, color="#303030")
    axes[2].annotate(r"$\beta=5$", (endpoint["CV"], endpoint["entropy_yield"]), xytext=(-10, -3), textcoords="offset points", ha="right", fontsize=8)
    for label, color, offset in (("evoked", "#d55e00", (12, -10)), ("spontaneous", "#0072b2", (-7, 14))):
        point = markers[label]
        axes[2].scatter(point["CV"], point["entropy_yield"], s=52, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        axes[2].annotate(f"{label.capitalize()} CV {MARKER_CVS[label]:.2f}", (point["CV"], point["entropy_yield"]), xytext=offset, textcoords="offset points", color=color, fontsize=8, ha="right" if label == "spontaneous" else "left")
    axes[2].set(title=r"C  V1 marker coordinates ($\alpha=0.7$)", xlabel="Coefficient of variation", ylabel=r"Model entropy yield $h/C$")
    for ax in axes[1:]:
        ax.axvline(1 / np.sqrt(3), color="#8b8b8b", lw=0.8, ls=":", zorder=0)
        ax.axvline(1, color="#8b8b8b", lw=0.8, ls=":", zorder=0)
    for ax in axes:
        ax.grid(axis="y", color="#e7e7e7", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Bounded power-cost reference: corrected continuum calculation", fontsize=13, y=0.965)
    fig.text(0.5, 0.89, r"$c(r)=0.37r^\alpha$, $0\leq r\leq100$; exact incomplete-gamma normalization and moments", ha="center", fontsize=10)
    fig.text(0.065, 0.085, r"Lines connect 80 nodes in $\beta$ order ($10^{-4}$ to 5); fixed differential-entropy units, not mutual information.", fontsize=8.5)
    fig.text(0.065, 0.04, "Panel C retains the V1 CV coordinates; vertical positions are model values, not empirical information. No interval-wide monotonicity claim.", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"FigureS12.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output in (ROOT / "FigS4", ROOT / "updated/figures"):
        raise ValueError("Use a separate output directory; legacy artwork is read-only")
    before = legacy_hashes()
    tests = validate()
    records = make_records()
    markers = {name: marker_state(cv) for name, cv in MARKER_CVS.items()}
    payload = {
        "method": "Exact truncated-power incomplete-gamma integrals evaluated in log space",
        "entropy_convention": "Differential entropy h=beta*C+logZ in fixed numerical response units; not categorical H or mutual information",
        "parameters": {"A": A_COST, "r_max": R_MAX, "alphas": ALPHAS, "betas": BETAS.tolist(), "panel_A_beta": 0.3},
        "formulas": {
            "I_j": "gamma_lower((j+1)/alpha, beta*A*R^alpha)/(alpha*(beta*A)^((j+1)/alpha))",
            "Z": "I_0", "moment_j": "I_j/I_0", "C": "A*I_alpha/I_0",
            "h": "beta*C+logZ", "yield_derivative": "Var(c)*logZ/C^2",
        },
        "scope": "Original 8-by-80 parameter grid. Derivative and ordering checks refer to finite nodes, not a proof on continuous CV intervals. Root solves locate the exact logZ=0 yield criterion; outside-sweep roots are flagged and not plotted.",
        "figure_labels": {"B_y": "Entropy yield h/C (nats / cost unit)", "C_y": "Model entropy yield h/C"},
        "records": records, "panel_C_markers": markers,
        "legacy_endpoint_comparison": legacy_comparison(records),
        "legacy_source_sha256": before,
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    }
    output.mkdir(parents=True, exist_ok=True)
    plot_figure(records, markers, output)
    tests["legacy_files_unchanged"] = before == legacy_hashes()
    if not tests["legacy_files_unchanged"]:
        raise AssertionError("Legacy source changed during audit")
    for filename, data in (("FigureS12.json", payload), ("FigureS12_tests.json", tests)):
        with (output / filename).open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
    for record in records:
        turn = record["yield_turn_logZ_zero"]
        print(f"alpha={record['alpha']:.1f}: beta_turn={turn['beta']:.9f}, CV_turn={turn['CV']:.9f}, negative_steps={record['negative_yield_steps']}/79, in_sweep={turn['inside_original_beta_sweep']}")
    print(f"PASS: adaptive quadrature max scaled error={tests['max_quad_scaled_absolute_error']:.3g}; legacy files unchanged")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
