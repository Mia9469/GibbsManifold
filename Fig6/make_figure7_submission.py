#!/usr/bin/env python3
"""Build data-driven scientific panels, never overlays on user Figure7 artwork.

The E/G/I names identify training, held-out, and structured SSIM panels.
The historical artwork layout may group G/H subpanels differently.
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "outputs/final_text_review_20260907/ann_corrected/panels"
TYPES = ["null", "cauchy", "cv_m"]
LABELS = ["Null", "SparseNet", "CVM"]
COLORS = ["#929292", "#287bb5", "#d7aa24"]


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def load_inputs(input_root):
    train = read_json(input_root / "Fig6/sparse_run_rows.json")
    stats = read_json(input_root / "Fig6/sparse_run_statistics.json")
    held = read_json(input_root / "Fig6/generalization_results/held_out_natural/summary.json")
    struct = read_json(input_root / "Fig6/generalization_results/structured_patterns/summary.json")
    bank = read_json(input_root / "Fig6/generalization_results/structured_patterns/input_manifest.json")
    if bank["n_unique_inputs"] != 30 or len(train) != 60 or len(held) != 30 or len(struct) != 60:
        raise ValueError("Incorrect corrected panel cohort")
    for typ, label in zip(TYPES, LABELS):
        for metric in ["SSIM", "ICE", "independence"]:
            mean = np.mean([r[metric] for r in train if r["type"] == typ])
            if abs(mean - stats["groups"][label][metric]["mean"]) > 1e-10:
                raise ValueError("Training rows and frozen group summaries disagree")
    return train, held, struct


def draw(ax, rows, metric, title, ylabel, letter="", paired=False, percent=False):
    scale = 100.0 if percent else 1.0
    values = [scale * np.asarray([r[metric] for r in rows if r["type"] == t])
              for t in TYPES]
    box = ax.boxplot(values, tick_labels=LABELS, widths=0.5,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "#222222", "linewidth": 1.2},
                     boxprops={"linewidth": 0.9})
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(values, 1):
        ax.scatter(i + rng.uniform(-0.07, 0.07, len(vals)), vals,
                   color="#444444", s=10, alpha=0.65, zorder=3)
        ax.scatter([i], [vals.mean()], marker="D", s=28,
                   facecolor="white", edgecolor="black", linewidth=0.7, zorder=4)
    if paired:
        folds = sorted({r["fold"] for r in rows})
        for fold in folds:
            by_type = {r["type"]: r for r in rows if r["fold"] == fold}
            ax.plot([1, 2, 3], [scale * by_type[t][metric] for t in TYPES],
                    color="#cccccc", linewidth=0.65, alpha=0.65, zorder=0)
    ax.set_title(title, fontsize=10, pad=9)
    ax.set_ylabel(ylabel)
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold")
    means = ", ".join(f"{v.mean():.4f}" for v in values)
    ax.text(0.5, -0.19, "Means: " + means, transform=ax.transAxes,
            ha="center", fontsize=8, color="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    return {label: {"mean": float(vals.mean()), "n": len(vals)}
            for label, vals in zip(LABELS, values)}


def save_figure(fig, directory, name):
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(directory / (name + ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def generate(input_root=ROOT, output_dir=DEFAULT_OUTPUT, only=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    train, held, struct = load_inputs(input_root)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    specs = [
        ("Figure7D_atom_decorrelation", train, "independence", "Training images; 20 runs",
         "Mean atom decorrelation (%)", "D", False, False),
        ("Figure7E_training_SSIM", train, "SSIM", "Training images; 20 runs",
         "SSIM (%)", "E", False, True),
        ("Figure7F_training_RAE", train, "ICE", "Training images; 20 runs",
         "RAE", "F", False, False),
        ("Figure7G_heldout_SSIM", held, "SSIM", "Held-out images; 10 paired folds",
         "SSIM (%)", "G", True, True),
        ("Figure7H_heldout_RAE", held, "ICE", "Held-out images; 10 paired folds",
         "RAE", "H", True, False),
        ("Figure7I_structured_SSIM", struct, "SSIM", "30 unique structured inputs; 20 runs",
         "SSIM (%)", "I", False, True),
        ("Figure7J_structured_RAE", struct, "ICE", "30 unique structured inputs; 20 runs",
         "RAE", "J", False, False),
        ("heldout_RE_control", held, "RE", "Held-out images; 10 paired folds",
         "Relative reconstruction error", "A", True, False),
        ("structured_RE_control", struct, "RE", "30 unique structured inputs; 20 runs",
         "Relative reconstruction error", "B", False, False),
    ]
    manifest = {"input_root": str(input_root), "user_artwork_modified": False,
                "SSIM_panel_mapping": {"E": "training", "G": "heldout", "I": "structured"},
                "points": "Runs or image folds; mean diamonds; boxes median/IQR, 1.5-IQR whiskers",
                "panels": {}}
    for name, rows, metric, title, ylabel, letter, paired, percent in specs:
        if only is not None and name != only:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        means = draw(ax, rows, metric, title, ylabel, letter, paired, percent)
        fig.tight_layout()
        save_figure(fig, output_dir, name)
        manifest["panels"][name] = {"metric": metric, "display_scale": 100 if percent else 1,
                                    "groups": means}
    if only is None:
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
        for ax, rows, title, letter, paired in zip(
                axes, [train, held, struct], ["Training: 20 runs", "Held-out: 10 folds",
                "Structured: 30 inputs, 20 runs"], ["E", "G", "I"], [False, True, False]):
            draw(ax, rows, "SSIM", title, "SSIM (%)", letter, paired, True)
        fig.tight_layout(w_pad=2.5)
        save_figure(fig, output_dir, "Figure7_SSIM_E_G_I")
        fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4))
        for row, data in enumerate([held, struct]):
            for col, (metric, ylabel) in enumerate([("RE", "Relative reconstruction error"),
                                                   ("SSIM", "SSIM (%)"), ("ICE", "RAE")]):
                draw(axes[row, col], data, metric,
                     "Held-out: 10 paired image folds" if row == 0 else "Structured: 30 inputs, 20 runs",
                     ylabel, chr(65+row*3+col), row == 0, metric == "SSIM")
        fig.tight_layout(w_pad=2.5, h_pad=3)
        save_figure(fig, output_dir, "matched_generalization_panels")
    with open(output_dir / ("panel_manifest.json" if only is None else only + "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    print(f"Scientific panels written to {output_dir}; source artwork untouched")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
