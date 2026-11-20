"""Standalone supplement-ready visualization; leaves manuscript artwork untouched."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/pdf/elife_one_second_g_20261111"
DATA = json.loads((HERE / "results.json").read_text())
KEY = "G_bits_per_spike"
COLORS = ["#4C91C9", "#299E7C", "#9662B8"]
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10, "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "axes.unicode_minus": False, "xtick.labelsize": 8.5,
})


def main():
    assert DATA["with_replay"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.35), sharey=True)
    fig.subplots_adjust(left=.086, right=.984, bottom=.20, top=.80, wspace=.19)
    groups = [
        ("repeat_bootstrap", "Repeat\nbootstrap", "percentile_95"),
        ("joint_seed_repeat_bootstrap", "Seed + repeat\nbootstrap", "percentile_95"),
        ("four_repeat_subset", "Four of eight\nrepeats", "percentile_95"),
        ("five_template_subset", "Five of ten\ntemplates", "range"),
    ]
    for x, (name, label, interval) in enumerate(groups):
        for j, color in enumerate(COLORS):
            d = DATA["sampling"][name][j][KEY]
            center, (lo, hi) = d["mean"] * 1000, np.array(d[interval]) * 1000
            a.errorbar(x + .20 * (j - 1), center, yerr=[[center - lo], [hi - center]],
                       color=color, fmt="o", ms=2.8, capsize=2, lw=.9)
    a.set_xticks(range(len(groups)), [v[1] for v in groups])
    controls = [
        ("original", "Original"),
        ("thinning", "Binomial\nthinning"),
        ("tilt", "Exponential\nreweighting"),
        ("replay_mixture", "Frozen replay\nmatched mixture"),
    ]
    for x, (control, label) in enumerate(controls):
        for j, (budget, color) in enumerate(zip(DATA["budget_order"], COLORS)):
            y = np.array([r["deltas"][KEY] for r in DATA["cases"]
                          if r["budget"] == budget and r["control"] == control]) * 1000
            assert len(y) == 8
            xx = x + .20 * (j - 1)
            b.scatter(xx + np.linspace(-.051, .051, 8), y, s=12, color=color, alpha=.57, lw=0)
            b.scatter(xx, y.mean(), s=24, marker="D", color=color, edgecolor="black", lw=.45, zorder=3)
    b.set_xticks(range(len(controls)), [v[1] for v in controls])
    for ax, panel, title in ((a, "A", "Evaluation sampling"), (b, "B", "Achieved-cost matching")):
        ax.set_title(title, pad=8)
        ax.text(0, 1.055, panel, transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.axhline(0, color=".55", lw=.65, ls=":", zorder=0)
        ax.set_xlim(-.44, 3.44)
        ax.set_ylim(-19, 12)
        ax.tick_params(length=3, width=.65)
    a.set_ylabel(r"Learned minus initial $G$ ($10^{-3}$ bits/spike)")
    handles = [Line2D([], [], ls="none", label="Target (spikes/s/cell):")]
    handles.extend(Line2D([], [], ls="none", marker="o", color=c, label=f"{b / 75:g}", ms=4.5)
                   for b, c in zip(DATA["budget_order"], COLORS))
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center",
               bbox_to_anchor=(.52, .985), handletextpad=.3, columnspacing=1.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "One_second_G_sensitivity.pdf")
    fig.savefig(HERE / "One_second_G_sensitivity.png", dpi=240)


if __name__ == "__main__":
    main()
