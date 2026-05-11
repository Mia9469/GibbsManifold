"""
v1_decoder_vs_CV.py
===================
Test whether decoder-based stimulus discriminability actually correlates with
neural variability (CV) on the macaque V1 dataset.

Motivation
----------
The framework predicts that trial-to-trial Poisson spike-count agreement
(P_agree = e^{-2lambda} I_0(2lambda)) rises monotonically with CV.  The
*reviewer-relevant* question is whether this regime-specific count-consistency
metric tracks anything that a downstream decoder would actually use.  If
high-CV neurons (or trials, or sub-populations) are more decodable, the
"reliability" interpretation gains empirical support beyond the Poisson
algebra; if not, the agreement metric is decoupled from decodable information
and the manuscript's caveats are vindicated.

Pipeline
--------
  Per-neuron analysis (the meaningful test):
    For each of the 159 V1 neurons:
      1. Compute per-trial spike count over the 1.28 s driven window.
      2. Compute the lifetime CV of those counts.
      3. Train an 8-class LDA on (counts -> stim) with stratified 10-fold CV.
      4. Record held-out accuracy and decoder-MI from confusion matrix.
    Then test:
      (a) Spearman correlation between per-neuron CV_L and per-neuron acc.
      (b) Spearman correlation between per-neuron CV_L and per-neuron MI.
      (c) Mean accuracy in CV_L quartiles (Q1 vs Q4).

  Sub-population analysis (sanity check at the population scale):
    Stratify the 159 neurons by CV_L into low/high halves; for each half
    fit a multivariate LDA on the 8-stim task and compare accuracy.  Note
    the full population is at the 100% ceiling, so this only matters as a
    cross-check.

Output
------
  v1_decoder_vs_CV_summary.json      summary statistics
  v1_decoder_vs_CV_scatter.pdf       per-neuron CV_L vs decoder accuracy
  v1_decoder_vs_CV_quartile_bar.pdf  CV_L quartile vs accuracy (mean ± SEM)
  v1_decoder_vs_CV_report.txt        short human-readable report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict


HERE     = Path(__file__).resolve().parent
MAT_PATH = HERE / "105l001p16.mat"
OUT_JSON = HERE / "v1_decoder_vs_CV_summary.json"
OUT_PDF1 = HERE / "v1_decoder_vs_CV_scatter.pdf"
OUT_PDF2 = HERE / "v1_decoder_vs_CV_quartile_bar.pdf"
OUT_TXT  = HERE / "v1_decoder_vs_CV_report.txt"

DRIVEN_BINS = 1280
N_V1        = 159
N_CLASSES   = 8
RNG_SEED    = 20260502
KFOLDS      = 10


# --- helpers ---------------------------------------------------------------

def load_v1_driven(mat_path: Path):
    print(f"[load] {mat_path}")
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    nd = m["neuralData"][0, 0]
    rasters = nd.spikeRasters
    stim    = nd.stim.ravel().astype(int)
    n = rasters.shape[0]

    X, y = [], []
    for i in range(n):
        r = rasters[i, 0]
        if r.shape[1] != DRIVEN_BINS or stim[i] == 0:
            continue
        counts = np.asarray(r.sum(axis=1)).ravel()
        X.append(counts)
        y.append(stim[i])
    X = np.asarray(X, dtype=float)        # (n_trials, 159)
    y = np.asarray(y, dtype=int)
    print(f"[load] driven trials: {len(y)}, V1 neurons: {X.shape[1]}, classes: {sorted(set(y))}")
    return X, y


def confusion_MI(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[t-1, p-1] += 1
    p = cm / cm.sum()
    p_t = p.sum(axis=1, keepdims=True)
    p_p = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(p > 0, np.log2(p / (p_t * p_p)), 0.0)
    mi = float(np.nansum(p * log_term))
    return mi, cm


def per_neuron_decode(X, y, k=KFOLDS, seed=RNG_SEED):
    """For each neuron, run a 1D 8-class LDA and return per-neuron acc + MI."""
    n_trials, n_neurons = X.shape
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = np.zeros(n_neurons)
    mis  = np.zeros(n_neurons)
    cv_l = np.zeros(n_neurons)
    fr   = np.zeros(n_neurons)

    print(f"[per-neuron] running 1D LDA on {n_neurons} neurons, {k}-fold CV...")
    for j in range(n_neurons):
        x = X[:, j:j+1]                          # (n_trials, 1)
        cv_l[j] = x.std(ddof=1) / x.mean() if x.mean() > 0 else np.nan
        fr[j]   = x.mean() / 1.28                # spikes/sec (1.28-s window)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        try:
            y_pred = cross_val_predict(clf, x, y, cv=skf, n_jobs=-1)
            accs[j] = (y_pred == y).mean()
            mis[j], _ = confusion_MI(y, y_pred, N_CLASSES)
        except Exception as e:
            accs[j] = np.nan; mis[j] = np.nan
        if j % 20 == 0:
            sys.stdout.write(f"\r  neuron {j+1}/{n_neurons}  median acc so far = {np.nanmedian(accs[:j+1])*100:.2f}%")
            sys.stdout.flush()
    print()
    return accs, mis, cv_l, fr


def subpop_decode(X, y, mask, label, k=KFOLDS, seed=RNG_SEED):
    """Multivariate LDA on a column-subset of neurons."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    Xs = X[:, mask]
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    y_pred = cross_val_predict(clf, Xs, y, cv=skf, n_jobs=-1)
    acc = float((y_pred == y).mean())
    mi, _ = confusion_MI(y, y_pred, N_CLASSES)
    print(f"  {label:>20s}  n={mask.sum():3d}  acc={acc*100:5.2f}%  MI={mi:.3f} bits")
    return {"label": label, "n_neurons": int(mask.sum()),
            "acc": acc, "MI_bits": mi}


def plot_scatter(cv_l, accs, fr, out_path):
    import matplotlib.pyplot as plt
    chance = 1.0 / N_CLASSES
    rho, pval = spearmanr(cv_l, accs, nan_policy="omit")

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
    sc = ax[0].scatter(cv_l, accs * 100, c=fr, cmap="viridis", s=18, edgecolor="none", alpha=0.85)
    ax[0].axhline(chance * 100, ls="--", color="grey", lw=0.8, label=f"chance = {chance*100:.1f}%")
    ax[0].set_xlabel("Per-neuron lifetime $\\mathrm{CV}_L$")
    ax[0].set_ylabel("8-stim decoder accuracy (%)")
    ax[0].set_title(f"Per-neuron decoder vs. $\\mathrm{{CV}}_L$\n"
                    f"Spearman $\\rho={rho:+.3f}$, $p={pval:.2g}$")
    cbar = plt.colorbar(sc, ax=ax[0]); cbar.set_label("mean firing rate (Hz)")
    ax[0].legend(loc="upper left", fontsize=8, frameon=False)

    ax[1].scatter(fr, accs * 100, s=18, color="#444", alpha=0.7)
    ax[1].axhline(chance * 100, ls="--", color="grey", lw=0.8)
    ax[1].set_xlabel("mean firing rate (Hz)")
    ax[1].set_ylabel("8-stim decoder accuracy (%)")
    rho2, p2 = spearmanr(fr, accs, nan_policy="omit")
    ax[1].set_title(f"Per-neuron decoder vs. firing rate\n"
                    f"Spearman $\\rho={rho2:+.3f}$, $p={p2:.2g}$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


def plot_quartile(cv_l, accs, out_path):
    import matplotlib.pyplot as plt
    chance = 1.0 / N_CLASSES
    qs = np.nanquantile(cv_l, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.digitize(cv_l, qs[1:-1])    # 0..3
    means, sems = np.zeros(4), np.zeros(4)
    for q in range(4):
        m = bins == q
        means[q] = np.nanmean(accs[m]) * 100
        sems[q]  = np.nanstd(accs[m]) / np.sqrt(np.sum(m)) * 100

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xs = np.arange(4)
    ax.bar(xs, means, yerr=sems, capsize=4, color=["#aac5e3", "#7ba2c4", "#4d7fa6", "#2a5d87"], edgecolor="k", lw=0.6)
    ax.axhline(chance * 100, ls="--", color="grey", lw=0.8, label=f"chance = {chance*100:.1f}%")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Q1\n(low CV)", "Q2", "Q3", "Q4\n(high CV)"])
    ax.set_ylabel("Per-neuron decoder accuracy (%)")
    ax.set_title("Decoder accuracy by $\\mathrm{CV}_L$ quartile (mean $\\pm$ SEM)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")
    return means, sems


# --- main ------------------------------------------------------------------

def main():
    X, y = load_v1_driven(MAT_PATH)
    classes = sorted(set(y.tolist()))
    n_classes = len(classes)
    chance = 1.0 / n_classes
    print(f"[info] {n_classes} classes; chance accuracy = {chance*100:.2f}%")

    # ---- per-neuron decoder ----
    accs, mis, cv_l, fr = per_neuron_decode(X, y)

    valid = np.isfinite(accs) & np.isfinite(cv_l)
    rho, pval = spearmanr(cv_l[valid], accs[valid])
    rho_mi, p_mi = spearmanr(cv_l[valid], mis[valid])
    rho_fr, p_fr = spearmanr(cv_l[valid], fr[valid])
    rho_acc_fr, p_acc_fr = spearmanr(fr[valid], accs[valid])

    # CV_L quartile means
    qs = np.nanquantile(cv_l[valid], [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.digitize(cv_l, qs[1:-1])
    quartile_means = []
    for q in range(4):
        m = (bins == q) & valid
        quartile_means.append({
            "quartile": q + 1,
            "n": int(m.sum()),
            "cv_l_range": [float(qs[q]), float(qs[q+1])],
            "acc_mean": float(np.nanmean(accs[m])),
            "acc_sem":  float(np.nanstd(accs[m]) / np.sqrt(m.sum())),
            "MI_mean":  float(np.nanmean(mis[m])),
            "fr_mean":  float(np.nanmean(fr[m])),
        })

    # ---- subpop decoder (high-CV vs low-CV halves) ----
    print("\n[sub-pop] split by median CV_L:")
    median_cv = float(np.nanmedian(cv_l[valid]))
    high_mask = (cv_l > median_cv) & valid
    low_mask  = (cv_l <= median_cv) & valid
    high_pop = subpop_decode(X, y, high_mask, "high-CV half")
    low_pop  = subpop_decode(X, y, low_mask,  "low-CV half")
    full_pop = subpop_decode(X, y, valid,     "full pop (all)")

    # ---- plots ----
    plot_scatter(cv_l, accs, fr, OUT_PDF1)
    means, sems = plot_quartile(cv_l, accs, OUT_PDF2)

    # ---- save summary ----
    summary = {
        "session": "105l001p16",
        "n_trials": int(len(y)),
        "n_neurons": int(X.shape[1]),
        "n_classes": int(n_classes),
        "chance_accuracy": chance,
        "per_neuron": {
            "n_valid": int(valid.sum()),
            "median_acc": float(np.nanmedian(accs[valid])),
            "mean_acc":   float(np.nanmean(accs[valid])),
            "median_MI":  float(np.nanmedian(mis[valid])),
            "median_cv_l": median_cv,
            "spearman_cv_vs_acc": {"rho": float(rho), "p": float(pval)},
            "spearman_cv_vs_MI":  {"rho": float(rho_mi), "p": float(p_mi)},
            "spearman_cv_vs_fr":  {"rho": float(rho_fr), "p": float(p_fr)},
            "spearman_fr_vs_acc": {"rho": float(rho_acc_fr), "p": float(p_acc_fr)},
            "quartiles": quartile_means,
        },
        "subpop": {"high": high_pop, "low": low_pop, "full": full_pop},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] wrote {OUT_JSON}")

    # ---- short report ----
    lines = [
        "=== V1 decoder-vs-CV report (session 105l001p16, n=159 V1 neurons) ===",
        "",
        f"Chance accuracy (8-class):                      {chance*100:.2f}%",
        f"Per-neuron decoder accuracy (median across 159): {summary['per_neuron']['median_acc']*100:.2f}%",
        f"Per-neuron decoder MI (median across 159):       {summary['per_neuron']['median_MI']:.3f} bits",
        f"  (max possible = log2(8) = 3.000 bits)",
        "",
        "--- Test of the reliability-CV link ---",
        f"Spearman rho(CV_L, accuracy) = {rho:+.3f}  (p = {pval:.2g})",
        f"Spearman rho(CV_L, MI)       = {rho_mi:+.3f}  (p = {p_mi:.2g})",
        f"Spearman rho(CV_L, FR)       = {rho_fr:+.3f}  (p = {p_fr:.2g})  [confound to interpret]",
        f"Spearman rho(FR,   accuracy) = {rho_acc_fr:+.3f}  (p = {p_acc_fr:.2g})  [confound to interpret]",
        "",
        "--- CV_L quartile means (mean +/- SEM) ---",
    ]
    for q in quartile_means:
        lines.append(
            f"  Q{q['quartile']} (CV_L in [{q['cv_l_range'][0]:.3f},{q['cv_l_range'][1]:.3f}], n={q['n']:3d}):  "
            f"acc = {q['acc_mean']*100:5.2f} +/- {q['acc_sem']*100:.2f}%   MI = {q['MI_mean']:.3f} bits   "
            f"FR = {q['fr_mean']:.2f} Hz"
        )
    lines += [
        "",
        "--- Sub-population multivariate LDA ---",
        f"  high-CV half (n={high_pop['n_neurons']:3d}): acc={high_pop['acc']*100:5.2f}%  MI={high_pop['MI_bits']:.3f} bits",
        f"  low-CV half  (n={low_pop['n_neurons']:3d}): acc={low_pop['acc']*100:5.2f}%  MI={low_pop['MI_bits']:.3f} bits",
        f"  full pool    (n={full_pop['n_neurons']:3d}): acc={full_pop['acc']*100:5.2f}%  MI={full_pop['MI_bits']:.3f} bits",
        "",
        "--- Interpretation key ---",
        "  rho(CV_L, acc) > 0  ->  high-CV neurons more decodable (supports reliability-CV link).",
        "  rho(CV_L, acc) < 0  ->  high-CV neurons LESS decodable (count agreement is silence-induced).",
        "  rho(CV_L, acc) ~ 0  ->  agreement metric decoupled from decodable info; current caveat justified.",
        "  Note: CV_L typically correlates negatively with FR, so the FR-vs-accuracy correlation",
        "  must be considered as a confound when interpreting the CV-vs-accuracy correlation.",
    ]
    OUT_TXT.write_text("\n".join(lines))
    print(f"[save] wrote {OUT_TXT}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
