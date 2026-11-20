"""V1 word-information projection and separate single-neuron count-readout audit.

LIF segments have different inputs and active plasticity. They receive no R0
anchor or continuous-entropy reference, and do not identify a learning effect.
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
from v1_original_rates import RATE_PROVENANCE, load_evoked_sessions, load_spontaneous


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Fig3.recompute_lif_information_audit import DEFAULT_OUTPUT_DIR, load_completed_audit
C0, C1 = 2.4, 0.37
R_REF = 1.0


def reference(cost: np.ndarray, r0: float) -> np.ndarray:
    rate = (np.asarray(cost, float) - C0) / C1
    return (r0 / np.log(2.0)) * (1.0 + np.log(rate / R_REF))


def reference_slope(cost: np.ndarray, r0: float) -> np.ndarray:
    return r0 / (np.log(2.0) * (np.asarray(cost, float) - C0))


def anchor_scale(fr: float, information: float) -> float:
    return float(information * np.log(2.0) / (1.0 + np.log(fr / R_REF)))


def load_v1() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sessions = load_evoked_sessions()
    fr = np.concatenate([s["fr"] for s in sessions])
    mi = np.concatenate([s["mi"] for s in sessions])
    spon = load_spontaneous()
    fr_spon, h_spon = spon["fr"], spon["hr"]
    keep = np.isfinite(fr) & np.isfinite(mi) & (fr > 0)
    keep_spon = np.isfinite(fr_spon) & np.isfinite(h_spon) & (fr_spon > 0)
    return fr[keep], mi[keep], fr_spon[keep_spon], h_spon[keep_spon]


def load_lif() -> tuple[dict, dict]:
    return load_completed_audit()


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.88", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    vfr, vmi, sfr, sh = load_v1()
    lif_meta, lif = load_lif()
    primary = int(np.flatnonzero(lif["windows_ms"] == 20)[0])
    ltime, lfr = lif["segment_end_s"], lif["firing_rate_hz"]
    lscore, lfano = lif["logscore_bits_s"][:, primary], lif["fano_bits_s"][:, primary]
    vc, sc, lc = C0 + C1 * vfr, C0 + C1 * sfr, C0 + C1 * lfr

    v1_anchor = int(np.argsort(vfr, kind="stable")[len(vfr) // 2])
    r0_v1 = anchor_scale(vfr[v1_anchor], vmi[v1_anchor])

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    grey, blue, orange = "#4d4d4d", "#377eb8", "#e6862f"

    # Count readouts stay separate from the continuous V1 reference.
    for row, divisor in enumerate((np.ones_like(lc), lc)):
        ax = axes[row, 0]
        ax.plot(lc, lscore / divisor, "o-", color=grey, markersize=3.2,
                linewidth=1.0, label="Held-out log-posterior, 20 ms")
        ax.plot(lc, lfano / divisor, "s--", color=blue, markersize=2.5,
                linewidth=1.0, label="Fano plug-in, 20 ms")
        for endpoint, label, offset in ((0, "12 s", (-22, 7)), (-1, "120 s", (6, 6))):
            ax.annotate(label, (lc[endpoint], lscore[endpoint] / divisor[endpoint]),
                        xytext=offset, textcoords="offset points")
        ax.axhline(0, color="0.6", lw=.5)
        ax.legend(frameon=False, loc="center right", fontsize=7)
    axes[0, 0].set_title("A  Single-neuron LIF audit\nDifferent input segments; no learning comparison")
    axes[0, 0].set_ylabel("Window-normalized readout (bits/s)")
    axes[1, 0].set_ylabel("Window-normalized readout / cost")

    # V1 column.
    x_v1 = np.linspace(max(C0 + C1 / np.e, min(vc.min(), sc.min())),
                       max(vc.max(), sc.max()) * 1.01, 500)
    y_v1 = reference(x_v1, r0_v1)
    axes[0, 1].scatter(sc, sh, s=9, alpha=0.40, color=blue, edgecolors="none",
                       label=r"spontaneous $H(W)$")
    axes[0, 1].scatter(vc, vmi, s=9, alpha=0.48, color=orange, edgecolors="none",
                       label=r"evoked $I(S,J;W)$")
    axes[0, 1].plot(x_v1, y_v1, color="black", linewidth=1.5,
                    label="calibrated reference")
    axes[0, 1].set_title(f"B  Macaque V1\n$R_0={r0_v1:.2f}$")
    axes[0, 1].set_ylabel("Entropy / information rate (bits/s)")
    axes[0, 1].legend(frameon=False, loc="upper right")

    axes[1, 1].scatter(vc, vmi / vc, s=9, alpha=0.48, color=orange, edgecolors="none",
                       label=r"evoked $I/\widetilde C$")
    axes[1, 1].plot(x_v1, y_v1 / x_v1, color="black", linewidth=1.5,
                    label="reference / cost")
    axes[1, 1].plot(x_v1, reference_slope(x_v1, r0_v1), "--", color="black",
                    linewidth=1.1, label="reference slope")
    axes[1, 1].set_ylabel("Rate per proxy cost")
    axes[1, 1].set_ylim(0, max(3.7, 1.1 * float(np.max(vmi / vc))))
    axes[1, 1].legend(frameon=False, loc="lower right")
    inset = axes[1, 1].inset_axes([0.55, 0.48, 0.42, 0.43])
    inset.scatter(sc, sh / sc, s=5, alpha=0.35, color=blue, edgecolors="none")
    inset.set_title(r"Spontaneous $H/\widetilde C$", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.spines[["top", "right"]].set_visible(False)

    for ax in axes.ravel():
        style_axis(ax)
        ax.set_xlabel(r"Affine cost proxy $\widetilde C$")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Figure5.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "estimands": {"V1": "I((S,J);W)", "LIF": "Fixed 20-ms one-neuron count log-posterior score and separate Fano plug-in"},
        "v1_rate_provenance": RATE_PROVENANCE,
        "LIF": {
            "R0": None, "anchor": None, "n_segments": int(lfr.size),
            "audit_run_id": lif_meta["run_id"], "primary_window_ms": 20,
            "scope": lif_meta["protocol"]["comparability"],
            "reference_omitted_reason": "Different count-readout estimand and unmatched input segments; no continuous-entropy calibration asserted",
            "physical_firing_rate_Hz_first_final": lfr[[0, -1]].tolist(),
            "physical_cost_first_final": lc[[0, -1]].tolist(),
            "logscore_bits_s_first_final": lscore[[0, -1]].tolist(),
            "logscore_per_cost_first_final": (lscore / lc)[[0, -1]].tolist(),
            "fano_per_cost_first_final": (lfano / lc)[[0, -1]].tolist(),
        },
        "V1": {
            "R0": r0_v1,
            "anchor": "median firing rate",
            "n_neurons": int(vfr.size),
            "anchor_firing_rate_Hz": float(vfr[v1_anchor]),
            "anchor_cost": float(vc[v1_anchor]),
            "anchor_information_bits_per_s": float(vmi[v1_anchor]),
            "firing_rate_Hz_range": [float(vfr.min()), float(vfr.max())],
            "cost_range": [float(vc.min()), float(vc.max())],
            "spontaneous_firing_rate_Hz_range": [float(sfr.min()), float(sfr.max())],
            "spontaneous_cost_range": [float(sc.min()), float(sc.max())],
            "fraction_below_reference": float(np.mean(vmi < reference(vc, r0_v1))),
        },
    }
    (output_dir / "figure5_corrected_anchors.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Wrote", out, out.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    main(parser.parse_args().output_dir)
