"""Draw the matched-sampling schematic for Supplementary Figure S5."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "updated" / "figures" / "FigureS5.pdf"


def panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(0.00, 1.02, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom")
    ax.text(0.08, 1.02, title, transform=ax.transAxes, fontsize=11,
            va="bottom")


def draw_raster(ax: plt.Axes) -> None:
    panel_label(ax, "A", "Matched marginal sampling")
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.4, 8.2)
    ax.axis("off")
    raster = np.array([
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    ])
    for row in range(raster.shape[0]):
        y = 6.6 - row
        ax.plot([1.5, 11.5], [y, y], color="0.86", lw=0.6)
        for col in np.flatnonzero(raster[row]):
            ax.vlines(1.8 + col, y - 0.30, y + 0.30, color="black", lw=1.2)
    ax.text(0.15, 5.65, "$S_1$", fontsize=10, ha="left")
    ax.text(0.15, 2.65, "$S_2$", fontsize=10, ha="left")
    ax.text(0.75, 4.2, "repeats", rotation=90, fontsize=9, va="center")
    for x, color, label in [(1.55, "#4daf4a", "$J_1$"), (4.55, "#984ea3", "$J_2$")]:
        ax.add_patch(Rectangle((x, 0.95), 3.55, 6.15, facecolor=color,
                               edgecolor=color, alpha=0.11, lw=1.2))
        ax.text(x + 1.78, 0.45, label, color=color, fontsize=10, ha="center")
    ax.text(6.6, -0.15, "All valid starts $J$ enter $P(W_T)$", fontsize=9.5, ha="center")


def draw_conditional(ax: plt.Axes) -> None:
    panel_label(ax, "B", "Condition on the same $(S,J)$")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.text(0.25, 6.9, "fixed $(S_1,J_1)$", fontsize=10)
    for i, word in enumerate(["0100", "0100", "0100"]):
        ax.text(1.0, 5.85 - 0.65 * i, word, family="monospace", fontsize=11)
    ax.text(4.1, 5.2, "$H(W_T\mid S_1,J_1)=0$", fontsize=10, color="#287a3f")
    ax.text(4.1, 4.45, "retained", fontsize=10, color="#287a3f", fontweight="bold")

    ax.text(0.25, 3.25, "fixed $(S_2,J_2)$", fontsize=10)
    for i, word in enumerate(["1001", "1010", "1001"]):
        ax.text(1.0, 2.2 - 0.65 * i, word, family="monospace", fontsize=11)
    ax.text(4.1, 1.55, "$H(W_T\mid S_2,J_2)>0$", fontsize=10, color="#7a3b8f")
    ax.add_patch(FancyArrowPatch((7.8, 5.05), (7.8, 2.2), arrowstyle="->",
                                 mutation_scale=12, color="0.35", lw=1.0))
    ax.text(8.15, 3.6, "weighted over\nall $(S,J)$", fontsize=9,
            va="center", ha="left")


def draw_estimand(ax: plt.Axes) -> None:
    panel_label(ax, "C", "Estimator and complementary decoder")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.text(5.0, 6.25,
            "$I((S,J);W_T)=H(W_T)-H(W_T\mid S,J)$",
            fontsize=12, ha="center")
    ax.text(5.0, 5.25, "same sliding starts in both entropy terms",
            fontsize=9.5, ha="center", color="0.25")
    ax.plot([0.8, 9.2], [4.3, 4.3], color="0.75", lw=0.8)
    ax.text(5.0, 3.35, "Independent-half count decoder", fontsize=11,
            ha="center", fontweight="bold")
    ax.text(5.0, 2.35, "predicts categorical stimulus identity $S$",
            fontsize=10, ha="center")
    ax.text(5.0, 1.25, "different response variable and scientific question",
            fontsize=9.5, ha="center", color="0.3")


def main() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25),
                             gridspec_kw={"width_ratios": [1.18, 1.0, 1.1]})
    draw_raster(axes[0])
    draw_conditional(axes[1])
    draw_estimand(axes[2])
    fig.tight_layout(w_pad=2.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
