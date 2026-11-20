"""Supplementary Figure S7: V1 reference-anchor sensitivity using original rates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_corrected_figure5 import anchor_scale, reference
from v1_original_rates import C0, C1, DEFAULT_OUTPUT_DIR, RATE_PROVENANCE, load_evoked_sessions


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    sessions = load_evoked_sessions()
    fr = np.concatenate([row["fr"] for row in sessions])
    information = np.concatenate([row["mi"] for row in sessions])
    keep = np.isfinite(fr) & np.isfinite(information) & (fr > 0)
    fr, information = fr[keep], information[keep]
    cost = C0 + C1 * fr
    anchors = {
        "Minimum rate": int(np.argmin(fr)),
        "Median rate": int(np.argsort(fr, kind="stable")[fr.size // 2]),
        "Maximum rate": int(np.argmax(fr)),
    }
    colors = {"Minimum rate": "C0", "Median rate": "C2", "Maximum rate": "C3"}
    scales = {name: anchor_scale(fr[index], information[index])
              for name, index in anchors.items()}
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "Times New Roman", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    cost_grid = np.linspace(max(C0 + C1 / np.e, cost.min() * 0.99),
                            cost.max() * 1.01, 400)
    for panel, ax in enumerate(axes):
        divisor = cost if panel else np.ones_like(cost)
        grid_divisor = cost_grid if panel else np.ones_like(cost_grid)
        ax.scatter(cost, information / divisor, s=8, c="0.5", alpha=0.4,
                   edgecolor="none", label="V1 data")
        for name, scale in scales.items():
            ax.plot(cost_grid, reference(cost_grid, scale) / grid_divisor,
                    lw=1.8, color=colors[name], label=f"{name} anchor: $R_0$={scale:.2f}")
            index = anchors[name]
            ax.plot(cost[index], information[index] / divisor[index], "o", ms=9,
                    mfc="none", mec=colors[name], mew=2)
        ax.set_xlabel(r"affine cost proxy $\widetilde C$")
        ax.legend(frameon=False, fontsize=8, loc="upper left" if panel == 0 else "upper right")
    axes[0].set_ylabel(r"Fine-time information rate $I(S,J;W)$ (bits/s)")
    axes[0].set_title(r"(A) $I$--$\widetilde C$ under three anchor choices")
    axes[1].set_ylabel(r"$I(S,J;W)/\widetilde C$")
    axes[1].set_title(r"(B) $I/\widetilde C$ under three anchor choices")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "FigureS7.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "v1_rate_provenance": RATE_PROVENANCE,
        "n_neurons": int(fr.size),
        "anchors": {
            name: {
                "R0": scale,
                "firing_rate_Hz": float(fr[anchors[name]]),
                "cost": float(cost[anchors[name]]),
                "information_bits_per_s": float(information[anchors[name]]),
                "median_information_gap_bits_per_s": float(np.median(information - reference(cost, scale))),
                "fraction_above_reference": float(np.mean(information >= reference(cost, scale))),
            }
            for name, scale in scales.items()
        },
    }
    (output_dir / "figureS7_anchor_sensitivity.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    main(parser.parse_args().output_dir)
