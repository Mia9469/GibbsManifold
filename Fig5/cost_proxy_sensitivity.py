"""Cost sensitivity for V1, a descriptive LIF count audit, and recurrent readouts.

The scan varies the affine proxy coefficients around the published defaults and
adds concave, convex, and saturating monotone signaling costs normalized at the
pooled median V1 firing rate.  It evaluates (i) the five-session V1 association
between lifetime CV and corrected I((S,J);W)/C after firing-rate control and
(ii) the fixed-20-ms single-neuron LIF held-out log-posterior score, and
(iii) the endpoint direction of the held-out recurrent-population decoder's
Fano information lower bound per cost.  The recurrent metric is kept separate
from the matched-word Direct-Method quantities because it is a population-level
decoder lower bound rather than a single-neuron information estimate. LIF
endpoints have different input segments; their contrast is not a learning effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from v1_original_rates import RATE_PROVENANCE, load_evoked_sessions


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from Fig3.recompute_lif_information_audit import DEFAULT_OUTPUT_DIR, load_completed_audit

C0_DEFAULT = 2.4
C1_DEFAULT = 0.37
C0_GRID = np.linspace(1.2, 4.8, 13)
C1_GRID = np.linspace(0.185, 0.74, 13)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    values = np.column_stack([x, y, z]).astype(float)
    values = values[np.all(np.isfinite(values), axis=1)]
    ranks = np.column_stack([rankdata(values[:, i]) for i in range(3)])
    control = np.column_stack([np.ones(ranks.shape[0]), ranks[:, 2]])
    rx = ranks[:, 0] - control @ np.linalg.lstsq(control, ranks[:, 0], rcond=None)[0]
    ry = ranks[:, 1] - control @ np.linalg.lstsq(control, ranks[:, 1], rcond=None)[0]
    return float(pearsonr(rx, ry).statistic)


def fisher_mean(rhos: list[float]) -> float:
    z = np.arctanh(np.clip(np.asarray(rhos, float), -1 + 1e-9, 1 - 1e-9))
    return float(np.tanh(z.mean()))


def load_v1() -> list[dict]:
    rows = []
    for session in load_evoked_sessions():
        fr, mi, cv = session["fr"], session["mi"], session["cv"]
        keep = np.isfinite(fr) & (fr > 0) & np.isfinite(mi) & np.isfinite(cv) & (cv > 0)
        rows.append({"session": session["session"], "fr": fr[keep], "mi": mi[keep], "cv": cv[keep]})
    return rows


def cost_function(fr: np.ndarray, kind: str, c0: float, c1: float, ref: float) -> np.ndarray:
    fr = np.asarray(fr, float)
    if kind == "affine":
        return c0 + c1 * fr
    if kind == "concave_sqrt":
        return c0 + c1 * ref * np.sqrt(fr / ref)
    if kind == "convex_1p5":
        return c0 + c1 * ref * np.power(fr / ref, 1.5)
    if kind == "saturating":
        # Matches the affine signaling cost at fr=ref and approaches a ceiling.
        return c0 + c1 * (2 * ref) * fr / (ref + fr)
    raise KeyError(kind)


def evaluate_v1(v1: list[dict], kind: str, c0: float, c1: float, ref: float) -> dict:
    rhos = []
    for row in v1:
        cost = cost_function(row["fr"], kind, c0, c1, ref)
        rhos.append(partial_spearman(row["cv"], row["mi"] / cost, row["fr"]))
    return {
        "per_session_rho": rhos,
        "fisher_mean_rho": fisher_mean(rhos),
        "sessions_positive": int(np.sum(np.asarray(rhos) > 0)),
    }


def load_lif_trajectory() -> dict:
    metadata, lif = load_completed_audit()
    primary = int(np.flatnonzero(lif["windows_ms"] == 20)[0])
    return {"time": lif["segment_end_s"], "mi": lif["logscore_bits_s"][:, primary],
            "fr": lif["firing_rate_hz"], "audit_run_id": metadata["run_id"],
            "scope": metadata["protocol"]["comparability"]}


def load_network_trajectory() -> dict[str, np.ndarray]:
    results = np.load(
        ROOT / "Fig4" / "network_information_corrected" / "network_information_results.npz"
    )
    windows_ms = np.asarray(results["decoder_windows_ms"], int)
    primary = int(np.flatnonzero(windows_ms == 20)[0])
    stages = np.asarray(results["stages"], float)
    # Checkpoints are saved after exponentially increasing training intervals.
    time = np.power(2.0, stages + 1.0) * 1e-4
    mi_lower_bound = np.asarray(results["decoder_mi_lower_bound_bits_s"], float)[:, primary]
    mean_rate = np.asarray(results["mean_rate"], float)
    keep = np.isfinite(time) & np.isfinite(mi_lower_bound) & np.isfinite(mean_rate) & (mean_rate > 0)
    return {
        "time": time[keep],
        "mi": mi_lower_bound[keep],
        "fr": mean_rate[keep],
        "window_ms": np.asarray(20),
    }


def evaluate_trajectory(row: dict[str, np.ndarray], kind: str, c0: float, c1: float, ref: float) -> dict:
    cost = cost_function(row["fr"], kind, c0, c1, ref)
    efficiency = row["mi"] / cost
    return {
        "spearman_time_vs_information_per_cost": float(spearmanr(row["time"], efficiency).statistic),
        "endpoint_delta_information_per_cost": float(efficiency[-1] - efficiency[0]),
        "spearman_cost_vs_information": float(spearmanr(cost, row["mi"]).statistic),
    }


def evaluate_lif(row: dict, kind: str, c0: float, c1: float, ref: float) -> dict:
    result = evaluate_trajectory(row, kind, c0, c1, ref)
    return {
        "spearman_time_vs_logscore_per_cost": result["spearman_time_vs_information_per_cost"],
        "endpoint_delta_logscore_per_cost": result["endpoint_delta_information_per_cost"],
        "spearman_cost_vs_logscore": result["spearman_cost_vs_information"],
    }


def make_figure(affine_cells: list[dict], output_dir: Path) -> None:
    """Render the coefficient scan without implying metabolic validation."""
    shape = (C0_GRID.size, C1_GRID.size)
    v1 = np.asarray([cell["v1"]["fisher_mean_rho"] for cell in affine_cells]).reshape(shape)
    lif = np.asarray([cell["lif"]["endpoint_delta_logscore_per_cost"] for cell in affine_cells]).reshape(shape)
    network = np.asarray([
        cell["network"]["endpoint_delta_information_per_cost"] for cell in affine_cells
    ]).reshape(shape)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.25), constrained_layout=True)
    panels = (
        (v1, r"A  Macaque V1", "Fisher-mean coefficient"),
        (lif, "B  LIF count audit (20 ms)\nDifferent input segments", r"endpoint $\Delta[I_q/\widetilde C]$"),
        (
            network,
            "C  Recurrent population",
            r"endpoint $\Delta[\mathrm{MI}_{\mathrm{LB}}/\widetilde C]$",
        ),
    )
    extent = [C1_GRID.min() / C1_DEFAULT, C1_GRID.max() / C1_DEFAULT,
              C0_GRID.min() / C0_DEFAULT, C0_GRID.max() / C0_DEFAULT]
    for ax, (values, title, colorbar_label) in zip(axes, panels):
        image = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        ax.scatter([1], [1], marker="x", s=46, linewidths=1.6, color="white", zorder=3)
        ax.set_xlabel(r"signaling coefficient $C_1/C_{1,0}$")
        ax.set_ylabel(r"baseline coefficient $C_0/C_{0,0}$")
        ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035, label=colorbar_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Figure_cost_proxy_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    v1 = load_v1()
    lif = load_lif_trajectory()
    network = load_network_trajectory()
    ref = float(np.median(np.concatenate([row["fr"] for row in v1])))

    affine_cells = []
    for c0 in C0_GRID:
        for c1 in C1_GRID:
            v1_result = evaluate_v1(v1, "affine", float(c0), float(c1), ref)
            lif_result = evaluate_lif(lif, "affine", float(c0), float(c1), ref)
            network_result = evaluate_trajectory(network, "affine", float(c0), float(c1), ref)
            affine_cells.append({
                "C0": float(c0),
                "C1": float(c1),
                "v1": v1_result,
                "lif": lif_result,
                "network": network_result,
            })

    alternatives = {}
    for kind in ("affine", "concave_sqrt", "convex_1p5", "saturating"):
        alternatives[kind] = {
            "v1": evaluate_v1(v1, kind, C0_DEFAULT, C1_DEFAULT, ref),
            "lif": evaluate_lif(lif, kind, C0_DEFAULT, C1_DEFAULT, ref),
            "network": evaluate_trajectory(network, kind, C0_DEFAULT, C1_DEFAULT, ref),
        }

    output = {
        "v1_rate_provenance": RATE_PROVENANCE,
        "lif_audit_run_id": lif["audit_run_id"],
        "lif_primary_window_ms": 20,
        "lif_readout": "Signed nested-repeat-held-out log-posterior I_q per 0.02 seconds; denominator uses original count/time firing rate",
        "lif_scope_note": lif["scope"],
        "partial_spearman_definition": "Pearson correlation of OLS residuals of ranks, controlling ranked original trial-count firing rate",
        "nonlinear_normalization_firing_rate_Hz": ref,
        "default_proxy": {"C0": C0_DEFAULT, "C1": C1_DEFAULT},
        "scan": {
            "C0_range": [float(C0_GRID.min()), float(C0_GRID.max())],
            "C1_range": [float(C1_GRID.min()), float(C1_GRID.max())],
            "n_parameter_pairs": len(affine_cells),
            "cells": affine_cells,
            "summary": {
                "pairs_with_v1_positive_in_all_five_sessions": int(sum(cell["v1"]["sessions_positive"] == 5 for cell in affine_cells)),
                "v1_fisher_mean_rho_range": [
                    float(min(cell["v1"]["fisher_mean_rho"] for cell in affine_cells)),
                    float(max(cell["v1"]["fisher_mean_rho"] for cell in affine_cells)),
                ],
                "pairs_with_positive_LIF_logscore_endpoint_delta": int(sum(cell["lif"]["endpoint_delta_logscore_per_cost"] > 0 for cell in affine_cells)),
                "pairs_with_negative_LIF_logscore_endpoint_delta": int(sum(cell["lif"]["endpoint_delta_logscore_per_cost"] < 0 for cell in affine_cells)),
                "LIF_logscore_endpoint_delta_range": [
                    float(min(cell["lif"]["endpoint_delta_logscore_per_cost"] for cell in affine_cells)),
                    float(max(cell["lif"]["endpoint_delta_logscore_per_cost"] for cell in affine_cells)),
                ],
                "pairs_with_positive_network_endpoint_delta": int(sum(cell["network"]["endpoint_delta_information_per_cost"] > 0 for cell in affine_cells)),
                "network_endpoint_delta_range": [
                    float(min(cell["network"]["endpoint_delta_information_per_cost"] for cell in affine_cells)),
                    float(max(cell["network"]["endpoint_delta_information_per_cost"] for cell in affine_cells)),
                ],
            },
        },
        "alternative_monotone_costs_at_default_normalization": alternatives,
        "normalization_note": (
            "Nonlinear signaling costs equal the default affine signaling term at the pooled median V1 firing rate. "
            "Because V1 inference also partials firing rate, stability across deterministic monotone costs is a denominator sensitivity check, not independent metabolic validation."
        ),
        "network_scope_note": (
            "The recurrent metric is the 20-ms held-out population decoder's Fano information "
            "lower bound per cost. It is analyzed separately from the single-neuron matched-word "
            "Direct-Method quantity and is not placed on the calibrated Fig. 5 reference curve."
        ),
        "R0_scope_note": (
            "R0 does not enter these empirical ratios. It only rescales the V1 reference in Fig. 5; "
            "the independent V1 anchor scan is reported in Fig. S7. No R0 anchor is assigned to the LIF count audit."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "cost_proxy_sensitivity.json"
    out_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    make_figure(affine_cells, output_dir)
    print(json.dumps(output["scan"]["summary"], indent=2))
    print(json.dumps(alternatives, indent=2))
    print("Wrote", out_json, output_dir / "Figure_cost_proxy_sensitivity.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    main(parser.parse_args().output_dir)
