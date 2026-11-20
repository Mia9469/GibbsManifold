"""Cross-validated controls for the V1 information-cost curve in Fig. 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from v1_original_rates import DEFAULT_OUTPUT_DIR, RATE_PROVENANCE, load_evoked_sessions

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

N_PERM = 1000
SEED = 20260620
MODELS = ("Gibbs (1 par.)", "log (2 par.)", "sqrt (2 par.)", "rate-linear (2 par.)")
COLORS = {
    "Gibbs (1 par.)": "#111111",
    "log (2 par.)": "#4C78A8",
    "sqrt (2 par.)": "#59A14F",
    "rate-linear (2 par.)": "#E15759",
}


def load_sessions() -> tuple[list[np.ndarray], list[np.ndarray]]:
    rates, info = [], []
    for session in load_evoked_sessions():
        r, y = session["fr"], session["mi"]
        keep = np.isfinite(r) & np.isfinite(y) & (r > 0)
        rates.append(r[keep])
        info.append(y[keep])
    return rates, info


def fit_predict(model: str, x: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if model == "Gibbs (1 par.)":
        basis = 1.0 + np.log(x)
        scale = max(0.0, float(np.dot(basis, y) / np.dot(basis, basis)))
        return scale * (1.0 + np.log(x_test))
    if model == "log (2 par.)":
        train = np.column_stack([np.ones(x.size), np.log(x)])
        test = np.column_stack([np.ones(x_test.size), np.log(x_test)])
    elif model == "sqrt (2 par.)":
        train = np.column_stack([np.ones(x.size), np.sqrt(x)])
        test = np.column_stack([np.ones(x_test.size), np.sqrt(x_test)])
    elif model == "rate-linear (2 par.)":
        train = np.column_stack([np.ones(x.size), x])
        test = np.column_stack([np.ones(x_test.size), x_test])
    else:
        raise ValueError(f"unknown model: {model}")
    coef = np.linalg.lstsq(train, y, rcond=None)[0]
    return test @ coef


def leave_one_session_out(
    rates: list[np.ndarray], info: list[np.ndarray], model: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed, predicted, fold_rmse = [], [], []
    for held_out in range(len(rates)):
        x_train = np.concatenate([x for i, x in enumerate(rates) if i != held_out])
        y_train = np.concatenate([y for i, y in enumerate(info) if i != held_out])
        pred = fit_predict(model, x_train, y_train, rates[held_out])
        observed.append(info[held_out])
        predicted.append(pred)
        fold_rmse.append(float(np.sqrt(np.mean((info[held_out] - pred) ** 2))))
    return np.concatenate(observed), np.concatenate(predicted), np.asarray(fold_rmse)


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def shuffled_null(
    rates: list[np.ndarray], info: list[np.ndarray], model: str, rng: np.random.Generator
) -> np.ndarray:
    values = np.empty(N_PERM, dtype=float)
    for rep in range(N_PERM):
        shuffled = [rng.permutation(y) for y in info]
        observed, predicted, _ = leave_one_session_out(rates, shuffled, model)
        values[rep] = r2_score(observed, predicted)
    return values


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    rates, info = load_sessions()
    rng = np.random.default_rng(SEED)
    observed_r2, fold_rmse, null_r2 = {}, {}, {}
    for model in MODELS:
        y, pred, rmse = leave_one_session_out(rates, info, model)
        observed_r2[model] = r2_score(y, pred)
        fold_rmse[model] = rmse
        null_r2[model] = shuffled_null(rates, info, model, rng)

    all_rate = np.concatenate(rates)
    all_info = np.concatenate(info)
    grid = np.linspace(all_rate.min(), all_rate.max(), 500)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))

    ax = axes[0]
    session_colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.9, len(rates)))
    for idx, (x, y) in enumerate(zip(rates, info)):
        ax.scatter(x, y, s=11, alpha=0.45, color=session_colors[idx],
                   edgecolor="none", label="V1 neurons (5 sessions)" if idx == 0 else None)
    for model in MODELS:
        ax.plot(grid, fit_predict(model, all_rate, all_info, grid),
                color=COLORS[model], lw=1.8, label=model)
    ax.set_xlabel("mean firing rate [Hz]")
    ax.set_ylabel("measured information [bits/s]")
    ax.set_title("A  V1 data and full-data fits")
    ax.legend(frameon=False, fontsize=6.8, ncol=2)

    x_pos = np.arange(len(MODELS))
    ax = axes[1]
    ax.bar(x_pos, [observed_r2[m] for m in MODELS],
           color=[COLORS[m] for m in MODELS], width=0.72)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x_pos, ["Gibbs", "log", "sqrt", "rate-\nlinear"])
    ax.set_ylabel(r"leave-one-session-out $R^2$")
    ax.set_title("B  Held-out-session prediction")
    for i, model in enumerate(MODELS):
        ax.text(i, observed_r2[model] + 0.025, f"{observed_r2[model]:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    parts = ax.violinplot([null_r2[m] for m in MODELS], positions=x_pos,
                          widths=0.78, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("0.75")
        body.set_edgecolor("0.5")
        body.set_alpha(0.8)
    for i, model in enumerate(MODELS):
        ax.scatter(i, observed_r2[model], s=48, color=COLORS[model],
                   edgecolor="black", linewidth=0.5, zorder=3)
        q95 = np.quantile(null_r2[model], 0.95)
        ax.plot([i - 0.22, i + 0.22], [q95, q95], color="0.25", lw=1.2)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x_pos, ["Gibbs", "log", "sqrt", "rate-\nlinear"])
    ax.set_ylabel(r"leave-one-session-out $R^2$")
    ax.set_title("C  Within-session permutation null")
    ax.text(0.03, 0.97, "colored points: observed\nbars: null 95th percentile",
            transform=ax.transAxes, fontsize=7.5, va="top")

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_fig = output_dir / "FigureS18.pdf"
    out_json = output_dir / "figureS18_frontier_controls.json"
    fig.savefig(out_fig, bbox_inches="tight")
    fig.savefig(out_fig.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "v1_rate_provenance": RATE_PROVENANCE,
        "historical_script_name": "figS20_frontier_model_controls.py",
        "current_supplement_figure": "S18",
        "seed": SEED,
        "n_permutations": N_PERM,
        "n_sessions": len(rates),
        "n_neurons": int(sum(len(x) for x in rates)),
        "models": {
            model: {
                "leave_one_session_out_r2": observed_r2[model],
                "fold_rmse_bits_per_s": fold_rmse[model].tolist(),
                "mean_fold_rmse_bits_per_s": float(fold_rmse[model].mean()),
                "null_r2_95th_percentile": float(np.quantile(null_r2[model], 0.95)),
                "permutation_p_upper": float(
                    (1 + np.sum(null_r2[model] >= observed_r2[model])) / (N_PERM + 1)
                ),
            }
            for model in MODELS
        },
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_fig}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    main(parser.parse_args().output_dir)
