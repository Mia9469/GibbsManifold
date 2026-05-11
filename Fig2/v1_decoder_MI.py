"""
v1_decoder_MI.py
================
Decoder-based stimulus discriminability and mutual information for the
macaque V1 dataset (CRCNS v1v2-1, session 105l001p16).

Per the reviewer:
  "increase a decoder analysis: compare stimulus identity decoding accuracy
   / cross-validated mutual information / Fisher information across CV
   regimes."

Pipeline
--------
  1. Load 105l001p16.mat -> spikeRasters (6400 trials x [V1, V2]).
  2. Restrict to driven trials (stim in {1..8}, 1280-bin rasters).
  3. Per trial, build a 159-dim feature vector x_t = spike count per V1
     neuron over the 1.28 s stimulus window.
  4. K-fold cross-validated decoders (k=10, stratified):
       - LDA (with shrinkage)
       - Linear SVM (one-vs-rest, C=1)
       - Multinomial logistic regression
  5. Report per-decoder accuracy (mean +/- SEM across folds), confusion
     matrix, and decoder-derived MI lower bound:
         I_dec(stim; pred) = H(stim) - H(stim | pred)
     computed from the confusion matrix.
  6. Permutation null: shuffle stim labels, re-run LDA across folds,
     report the chance-level accuracy and MI distribution (1000 reps).
  7. Save a JSON summary + a confusion-matrix PDF.

Output
------
  /Users/mia/Library/Mobile Documents/com~apple~CloudDocs/GitHub/GibbsManifold/Fig2/
    v1_decoder_summary.json
    v1_decoder_confusion.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC


HERE = Path(__file__).resolve().parent
MAT_PATH = HERE / "105l001p16.mat"
OUT_JSON = HERE / "v1_decoder_summary.json"
OUT_PDF  = HERE / "v1_decoder_confusion.pdf"

DRIVEN_BINS = 1280       # bins/trial that mark a driven trial
N_V1        = 159
RNG_SEED    = 20260427
N_PERM      = 1000


# --- helpers ---------------------------------------------------------------

def load_v1_driven(mat_path: Path):
    print(f"[load] {mat_path}")
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    nd = m["neuralData"][0, 0]
    rasters = nd.spikeRasters       # shape (6400, 2)
    stim    = nd.stim.ravel().astype(int)
    n = rasters.shape[0]

    X = []   # (n_trials, n_v1)
    y = []   # stim id
    for i in range(n):
        r = rasters[i, 0]   # V1 raster
        if r.shape[1] != DRIVEN_BINS:
            continue
        if stim[i] == 0:
            continue            # spontaneous
        # rasters are scipy sparse csc; .sum returns np.matrix -> flatten
        counts = np.asarray(r.sum(axis=1)).ravel()
        X.append(counts)
        y.append(stim[i])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    print(f"[load] driven trials: {len(y)}, V1 neurons: {X.shape[1]}, classes: {sorted(set(y))}")
    return X, y


def confusion_MI(y_true, y_pred, n_classes):
    """Mutual information (bits) between true label and predicted label,
    estimated from the confusion matrix."""
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


def run_decoder(name, clf, X, y, n_classes, k=10, seed=RNG_SEED):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    y_pred = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)
    mi, cm = confusion_MI(y, y_pred, n_classes)
    print(f"  {name:>20s}   acc = {accs.mean()*100:5.2f} +/- {accs.std()*100:.2f}%   MI = {mi:.3f} bits")
    return {
        "name": name,
        "fold_accs": accs.tolist(),
        "acc_mean": float(accs.mean()),
        "acc_sem":  float(accs.std() / np.sqrt(len(accs))),
        "MI_bits":  mi,
        "confusion": cm.tolist(),
    }


def chance_baseline(X, y, n_classes, n_perm=N_PERM, k=10, seed=RNG_SEED):
    """Permute labels, run LDA k-fold, record acc + MI."""
    rng = np.random.default_rng(seed)
    accs, mis = np.zeros(n_perm), np.zeros(n_perm)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    print(f"[null] {n_perm} label permutations of LDA ({k}-fold)...")
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        a = cross_val_score(clf, X, y_shuf, cv=skf, scoring="accuracy", n_jobs=1).mean()
        accs[i] = a
        if i % 50 == 0:
            sys.stdout.write(f"\r  perm {i+1}/{n_perm}   acc so far = {accs[:i+1].mean()*100:.2f}%")
            sys.stdout.flush()
    print()
    p_acc_null = float(accs.mean())
    return {
        "n_perm": n_perm,
        "acc_mean_null": p_acc_null,
        "acc_std_null": float(accs.std()),
        "acc_quantiles_null": [float(q) for q in np.quantile(accs, [0.025, 0.5, 0.975])],
    }


def upper_bound_info(n_classes, n_trials_per_class):
    """Theoretical info bounds for context."""
    return {
        "stim_entropy_bits": float(np.log2(n_classes)),
        "n_classes": n_classes,
    }


def plot_confusion(cm, n_classes, out_path: Path, title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.2))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="viridis", aspect="equal")
    for i in range(n_classes):
        for j in range(n_classes):
            txt_color = "w" if cm_norm[i, j] < 0.5 else "k"
            ax.text(j, i, f"{cm_norm[i, j]*100:.0f}", ha="center", va="center",
                    color=txt_color, fontsize=8)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(range(1, n_classes+1))
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(range(1, n_classes+1))
    ax.set_xlabel("Predicted stim")
    ax.set_ylabel("True stim")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="P(pred | true)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


# --- main ------------------------------------------------------------------

def main():
    X, y = load_v1_driven(MAT_PATH)
    classes = sorted(set(y.tolist()))
    n_classes = len(classes)
    print(f"[info] {n_classes} classes; max possible MI = log2({n_classes}) = "
          f"{np.log2(n_classes):.3f} bits")

    print("\n[decoders]")
    decoders = [
        ("LDA (shrinkage)", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ("Linear SVM",      LinearSVC(C=1.0, dual="auto", max_iter=5000)),
        ("MultinomLogistic", LogisticRegression(multi_class="multinomial",
                                                 solver="lbfgs", max_iter=1000,
                                                 n_jobs=-1)),
    ]

    results = []
    for name, clf in decoders:
        results.append(run_decoder(name, clf, X, y, n_classes))

    # Chance baseline
    null = chance_baseline(X, y, n_classes, n_perm=N_PERM)
    print(f"\n[null] chance accuracy = {null['acc_mean_null']*100:.2f}%  "
          f"(95% range [{null['acc_quantiles_null'][0]*100:.2f}, "
          f"{null['acc_quantiles_null'][2]*100:.2f}])")

    # Plot confusion for the best decoder
    best = max(results, key=lambda r: r["acc_mean"])
    cm = np.asarray(best["confusion"])
    plot_confusion(cm, n_classes, OUT_PDF,
                   f"V1 decoder confusion (best = {best['name']})")

    summary = {
        "session": "105l001p16",
        "n_trials": int(len(y)),
        "n_classes": n_classes,
        "n_v1_neurons": int(X.shape[1]),
        "feature": "spike count per V1 neuron over 1.28 s driven window",
        "max_possible_MI_bits": float(np.log2(n_classes)),
        "decoders": results,
        "null": null,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] wrote {OUT_JSON}")

    print("\n=== SUMMARY ===")
    print(f"max possible MI (uniform over {n_classes} stimuli): "
          f"{summary['max_possible_MI_bits']:.3f} bits")
    print(f"chance accuracy:                                   "
          f"{null['acc_mean_null']*100:.2f}%")
    for r in results:
        gain = (r["acc_mean"] - null["acc_mean_null"]) * 100
        print(f"  {r['name']:>20s}: acc={r['acc_mean']*100:5.2f}%  "
              f"(+{gain:.2f} over chance)   MI={r['MI_bits']:.3f} bits  "
              f"({100*r['MI_bits']/summary['max_possible_MI_bits']:.1f}% of ceiling)")


if __name__ == "__main__":
    main()
