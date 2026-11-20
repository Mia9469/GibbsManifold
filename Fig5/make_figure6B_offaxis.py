from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import gammaln


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "Figure6B_shape_CV_offaxis.pdf"


def cv_of_alpha(alpha):
    alpha = np.asarray(alpha, dtype=float)
    log_ratio = (
        gammaln(3.0 / alpha)
        + gammaln(1.0 / alpha)
        - 2.0 * gammaln(2.0 / alpha)
    )
    return np.sqrt(np.maximum(np.exp(log_ratio) - 1.0, 0.0))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10.5,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
        }
    )

    spontaneous_alpha = 2.99
    evoked_alpha = 8.00
    spontaneous_color = "#3373D9"
    evoked_color = "#FF8C33"

    fig, ax = plt.subplots(figsize=(3.35, 2.80))
    alpha = np.linspace(0.20, 3.10, 500)
    ax.plot(alpha, cv_of_alpha(alpha), color="#3F3F3F", lw=1.8, zorder=2)
    ax.axhline(1.0, color="#9A9A9A", lw=0.8, ls=(0, (2, 2)), zorder=1)

    spontaneous_cv = float(cv_of_alpha(spontaneous_alpha))
    evoked_cv = float(cv_of_alpha(evoked_alpha))
    ax.scatter(
        spontaneous_alpha,
        spontaneous_cv,
        s=72,
        color=spontaneous_color,
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )

    # The evoked median lies outside the displayed alpha range. The arrow
    # encodes direction only and avoids plotting alpha=8 at alpha=3.
    ax.annotate(
        "",
        xy=(3.17, evoked_cv),
        xytext=(2.66, evoked_cv),
        arrowprops=dict(
            arrowstyle="-|>",
            color=evoked_color,
            lw=1.8,
            mutation_scale=13,
        ),
        annotation_clip=False,
        zorder=5,
    )

    ax.annotate(
        "exponential\n(linear cost)",
        xy=(1.0, 1.0),
        xytext=(1.48, 1.42),
        ha="left",
        va="center",
        fontsize=9.2,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=spontaneous_color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=7.8,
            label=r"spontaneous: $\alpha \approx 2.99$",
        ),
        Line2D(
            [0],
            [0],
            marker=">",
            linestyle="none",
            markerfacecolor=evoked_color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=8.2,
            label=r"evoked: $\alpha \approx 8.00$ (off axis)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=8.7,
        handletextpad=0.55,
        borderaxespad=0.15,
        labelspacing=0.40,
    )

    ax.set_xlim(0.20, 3.10)
    ax.set_ylim(0.48, 4.02)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax.set_xlabel(r"effective shape $\alpha$", labelpad=4)
    ax.set_ylabel(r"distributional $\mathrm{CV}(\alpha)$", labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3.5, pad=2.5)

    fig.text(0.015, 0.975, "B", ha="left", va="top", fontsize=17)
    fig.subplots_adjust(left=0.20, right=0.955, bottom=0.20, top=0.94)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
