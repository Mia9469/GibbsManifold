"""Vector figures for the theory and controlled-budget results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/overleaf/GibbsManifold_eLife_Editor_Focus_20261111/figures"
COLORS = ["#4C91C9", "#299E7C", "#9662B8"]
BUDGETS = (240, 480, 720)
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 9.5, "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
    "axes.unicode_minus": False, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})


def heading(ax, letter, title):
    ax.text(-.15, 1.12, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="center")
    ax.set_title(title, pad=10)
    ax.tick_params(length=2.7, width=.6)


def theory():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(8.4, 2.55))
    fig.subplots_adjust(left=.085, bottom=.23, right=.98, top=.81, wspace=.49)
    k = np.linspace(.012, 9, 550)
    mu = 1 / k - 1 / np.expm1(k)
    var = 1 / k**2 - np.exp(-k) / (-np.expm1(-k))**2
    entropy = np.log(-np.expm1(-k) / k) + k * mu
    a.plot(np.r_[.5, mu], np.r_[0, entropy], color="#174F69", lw=2)
    a.fill_between(np.r_[.5, mu], np.r_[0, entropy], -1.35, color="#E1EDF2")
    a.text(.27, -.78, "Below the\nfrontier", ha="center", color="#174F69")
    a.set(xlim=(.1, .52), ylim=(-1.32, .1), xlabel=r"Mean rate, $C/r_{\max}$",
          ylabel=r"Maximum entropy, $H^\star$ (nats)")
    b.plot(np.r_[.5, mu], np.r_[1/np.sqrt(3), np.sqrt(var) / mu], color="#174F69", lw=2)
    b.annotate("Tighter budget", xy=(.13, .66), xytext=(.48, .66), ha="right", va="bottom",
               arrowprops=dict(arrowstyle="->", color="#444", lw=.8), fontsize=8.5)
    b.set(xlim=(.1, .52), ylim=(.54, 1.04), xlabel=r"Mean rate, $C/r_{\max}$",
          ylabel="CV of the optimizer")
    c.set(xlim=(0, 1), ylim=(0, 1))
    c.axis("off")
    c.text(.30, .89, "Together", ha="center")
    c.text(.77, .89, "Alternating", ha="center")
    for start, matrix in [(0.13, [[0, 1], [0, 1]]), (.60, [[0, 1], [1, 0]])]:
        for row in range(2):
            for col in range(2):
                value = matrix[row][col]
                x, y = start + .173 * col, .57 - row * .235
                c.add_patch(Rectangle((x, y), .165, .22, facecolor="#174F69" if value else "#E1EDF2",
                                      lw=0))
                c.text(x + .0825, y + .11, str(value), color="white" if value else "#222",
                       ha="center", va="center")
    c.text(.08, .68, "1", ha="center", va="center", fontsize=8)
    c.text(.08, .445, "2", ha="center", va="center", fontsize=8)
    c.text(.52, .15, "Same counts after pooling", ha="center", fontsize=8.5)
    c.text(.52, .02, "Different simultaneous patterns", ha="center", fontsize=8.5)
    for ax, letter, title in zip((a, b, c), "ABC", ("Entropy--cost frontier", "CV on the frontier", "Pooling loses structure")):
        heading(ax, letter, title)
    fig.savefig(OUT / "Figure_theory_focus.pdf")
    plt.close(fig)


def budget():
    with (HERE / "counts.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    bridge = json.loads((ROOT / "outputs/elife_reconsideration_20261109/analysis/results.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.65))
    fig.subplots_adjust(left=.09, right=.985, top=.855, bottom=.105, wspace=.53, hspace=.60)
    a, b, c, d, e, f = axes.flat

    def series(target, group, window, key):
        selected = [r for r in records if int(r["budget"]) == target and
                    int(r["window"]) == window and r["group"] == group]
        selected.sort(key=lambda r: (int(r["seed"]), int(r["stage"])))
        return np.array([float(r[key]) for r in selected]).reshape(8, 5)

    def points(ax, x, y, color):
        ax.errorbar(x, y.mean(axis=0), yerr=y.std(axis=0, ddof=1),
                    color=color, marker="o", markersize=3, lw=.85, capsize=2)

    stages = np.arange(5)
    windows = [20, 50, 100, 250, 500, 1000]
    handles = []
    for j, (target, color) in enumerate(zip(BUDGETS, COLORS)):
        rate = series(target, "all", 1000, "target_fraction")
        re = series(target, "all", 1000, "RE")
        g = series(target, "all", 1000, "G")
        points(a, stages, rate, color)
        points(b, stages, re, color)
        re_diff, g_diff = re[:, -1] - re[:, 0], g[:, -1] - g[:, 0]
        c.scatter(g_diff, re_diff, s=11, alpha=.63, color=color, linewidths=0)
        c.scatter(g_diff.mean(), re_diff.mean(), s=21, marker="D", color=color, edgecolor="#222", lw=.5)
        window_g = np.stack([series(target, "all", w, "G") for w in windows])
        points(d, np.arange(6) + (j - 1) * .05, (window_g[:, :, -1] - window_g[:, :, 0]).T, color)
        by_type = np.stack([series(target, group, 1000, "G") for group in ("all", "excitatory", "inhibitory")])
        x = np.arange(3) + (j - 1) * .22
        by_type = by_type[:, :, -1] - by_type[:, :, 0]
        for pos, ys in zip(x, by_type):
            e.scatter(pos + np.linspace(-.046, .046, 8), ys, s=10, alpha=.55, color=color, linewidths=0)
        e.scatter(x, by_type.mean(axis=1), s=20, marker="D", color=color, edgecolor="#222", lw=.5)
        gap = next(item for item in bridge["count_results"] if item["budget"] == target)
        marginal = np.array(gap["metrics"]["mean_marginal_gap_per_spike"]["seed_deltas"])
        f.scatter(g_diff, marginal, s=11, alpha=.63, color=color, linewidths=0)
        f.scatter(g_diff.mean(), marginal.mean(), s=21, marker="D", color=color, edgecolor="#222", lw=.5)
        handles.append(plt.Line2D([], [], color=color, marker="o", lw=1, ms=4, label=f"{target/75:g}"))
    a.axhspan(.95, 1.05, color=".93", zorder=-2)
    a.axhline(1, color=".5", ls=":", lw=.7)
    a.set(ylim=(.944, 1.056), ylabel="Achieved / target")
    for ax in (a, b):
        ax.set_xticks(stages, ("0", "26", "105", "419", "1678"))
        ax.set_xlabel("Training time (s; saved stages)")
    b.set_ylabel("Reconstruction error")
    c.set(xlabel=r"One-second $\Delta G$ (bits/spike)", ylabel="Change in reconstruction error")
    d.set_xticks(np.arange(6), windows)
    d.set(xlabel="Count window (ms)", ylabel=r"$\Delta G$ (bits/spike)")
    d.axvspan(4.75, 5.25, color=".92", zorder=-2)
    e.set_xticks(np.arange(3), ["E + I", "E only", "I only"])
    e.set_ylabel(r"One-second $\Delta G$ (bits/spike)")
    f.set(xlabel=r"Pooled $\Delta G$ (bits/spike)", ylabel="Change in mean marginal\ngap (bits/spike)")
    for ax in (c, f):
        ax.axvline(0, color=".55", ls=":", lw=.65)
    for ax in (c, d, e, f):
        ax.axhline(0, color=".55", ls=":", lw=.65)
    for ax, letter, title in zip(axes.flat, "ABCDEF", ("Budget control", "Held-out reconstruction",
            "Task and pooled gap", "Window dependence", "Cell-type comparison", "Cell identity matters")):
        heading(ax, letter, title)
    fig.legend(handles=handles, title="Target (spikes/s/cell)", ncol=3, loc="upper center",
               bbox_to_anchor=(.52, .994), frameon=False)
    fig.savefig(OUT / "Figure_budget_focus.pdf")
    plt.close(fig)


if __name__ == "__main__":
    theory()
    budget()
