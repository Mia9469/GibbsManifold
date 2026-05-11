"""
v1_decoder_vs_CV_panels.py
==========================
Generate a 2-panel figure summarizing the FR-controlled decoder analysis.

Panel A: scatter of per-neuron CV_L vs decoder accuracy, color = FR.
Panel B: within-FR-bin Spearman rho (CV_L vs accuracy), one bar per FR bin.

Reads from the JSON written by v1_decoder_vs_CV_partial.py and the per-neuron
arrays recomputed inline (cheap to redo).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict


HERE     = Path(__file__).resolve().parent
MAT_PATH = HERE / "105l001p16.mat"
JSON_IN  = HERE / "v1_decoder_vs_CV_partial.json"
OUT_PDF  = HERE / "v1_decoder_vs_CV_FR_controlled.pdf"

DRIVEN_BINS = 1280
N_CLASSES   = 8
RNG_SEED    = 20260502
KFOLDS      = 10


def load_v1_driven(mat_path: Path):
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    nd = m["neuralData"][0, 0]
    rasters = nd.spikeRasters
    stim    = nd.stim.ravel().astype(int)
    X, y = [], []
    for i in range(rasters.shape[0]):
        r = rasters[i, 0]
        if r.shape[1] != DRIVEN_BINS or stim[i] == 0:
            continue
        X.append(np.asarray(r.sum(axis=1)).ravel())
        y.append(stim[i])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def per_neuron_decode(X, y, k=KFOLDS, seed=RNG_SEED):
    n_trials, n_neurons = X.shape
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = np.full(n_neurons, np.nan)
    cv_l = np.full(n_neurons, np.nan)
    fr   = np.full(n_neurons, np.nan)
    for j in range(n_neurons):
        x = X[:, j:j+1]
        m = x.mean()
        if m <= 0:
            continue
        cv_l[j] = x.std(ddof=1) / m
        fr[j]   = m / 1.28
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        try:
            y_pred = cross_val_predict(clf, x, y, cv=skf, n_jobs=-1)
            accs[j] = (y_pred == y).mean()
        except Exception:
            pass
    return accs, cv_l, fr


def main():
    summary = json.loads(JSON_IN.read_text())
    X, y = load_v1_driven(MAT_PATH)
    accs, cv_l, fr = per_neuron_decode(X, y)
    valid = np.isfinite(accs) & np.isfinite(cv_l) & np.isfinite(fr)
    accs, cv_l, fr = accs[valid], cv_l[valid], fr[valid]
    chance = 1.0 / N_CLASSES

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))

    # Panel A: scatter colored by FR
    sc = ax[0].scatter(cv_l, accs * 100, c=fr, s=28, cmap="viridis",
                       edgecolor="white", linewidth=0.4)
    ax[0].axhline(chance * 100, ls="--", color="grey", lw=0.8,
                  label=f"chance ({chance*100:.1f}%)")
    rho_raw = summary["raw_spearman"]["cv_vs_acc"]["rho"]
    p_raw   = summary["raw_spearman"]["cv_vs_acc"]["p"]
    rho_par = summary["partial_spearman"]["cv_vs_acc_given_fr"]["rho"]
    p_par   = summary["partial_spearman"]["cv_vs_acc_given_fr"]["p"]
    ax[0].set_xlabel("Per-neuron lifetime $\\mathrm{CV}_L$")
    ax[0].set_ylabel("Per-neuron 8-stim decoder accuracy (%)")
    ax[0].set_title(
        f"Per-neuron decoder vs $\\mathrm{{CV}}_L$\n"
        f"$\\rho_{{\\mathrm{{raw}}}}={rho_raw:+.2f}$ (p={p_raw:.0e}),  "
        f"$\\rho_{{\\,|\\,\\mathrm{{FR}}}}={rho_par:+.2f}$ (p={p_par:.0e})"
    )
    cbar = plt.colorbar(sc, ax=ax[0]); cbar.set_label("mean firing rate (Hz)")
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")

    # Panel B: within-FR-bin rho
    bins = summary["fr_quartile_bins"]
    xs = np.arange(len(bins))
    rhos = [b["cv_acc_rho"] for b in bins]
    ps   = [b["cv_acc_p"]   for b in bins]
    fr_lo = [b["fr_range"][0] for b in bins]
    fr_hi = [b["fr_range"][1] for b in bins]
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(bins)))
    bar = ax[1].bar(xs, rhos, color=colors, edgecolor="k", lw=0.6)
    for i, (r, p) in enumerate(zip(rhos, ps)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax[1].text(i, r + 0.04, f"{sig}\np={p:.0e}", ha="center", fontsize=8)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"Q{i+1}\n{lo:.1f}-{hi:.1f}\nHz"
                           for i, (lo, hi) in enumerate(zip(fr_lo, fr_hi))],
                          fontsize=8)
    ax[1].set_ylabel("Spearman $\\rho(\\mathrm{CV}_L, \\mathrm{accuracy})$ within FR bin")
    ax[1].set_xlabel("Firing-rate quartile (FR control)")
    ax[1].set_title("Within-FR-bin CV-decodability link")
    ax[1].set_ylim(-0.2, 1.1)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
