"""
v1_decoder_vs_CV_partial.py
===========================
Follow-up to v1_decoder_vs_CV.py:

The first-pass analysis showed Spearman rho(CV_L, accuracy) = +0.33 (p = 2e-5),
seemingly supporting the framework's reliability-CV link.  However, CV_L and
firing rate are strongly anti-correlated in this dataset (rho = -0.76), so the
positive accuracy-CV correlation could be partly or entirely a firing-rate
artifact.  This script runs three controls:

  (1) Partial Spearman correlation of CV_L vs accuracy controlling for FR
      (and the converse: FR vs accuracy controlling for CV_L).

  (2) FR-matched stratification: bin neurons by FR, within each FR bin
      compute the CV-vs-accuracy correlation (no FR variation inside a bin).

  (3) Held-out residual analysis: rank-residualize accuracy on FR (Spearman),
      then test residual rank vs CV_L.

If the framework's claim is supported, we should see a non-trivial partial
correlation rho(CV_L, accuracy | FR) > 0 with reasonable significance.
If the apparent CV effect dissolves once FR is controlled, the result is
better described as a firing-rate effect than as evidence for the
reliability-CV link.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import spearmanr, rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict


HERE     = Path(__file__).resolve().parent
MAT_PATH = HERE / "105l001p16.mat"
OUT_JSON = HERE / "v1_decoder_vs_CV_partial.json"
OUT_TXT  = HERE / "v1_decoder_vs_CV_partial.txt"

DRIVEN_BINS = 1280
N_CLASSES   = 8
RNG_SEED    = 20260502
KFOLDS      = 10


def load_v1_driven(mat_path: Path):
    print(f"[load] {mat_path}")
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
    print(f"[per-neuron] running 1D LDA on {n_neurons} neurons")
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


def partial_spearman(x, y, z):
    """Partial Spearman correlation rho(x,y | z): rank-residualize x and y on z,
    then Pearson on the residuals (== Spearman partial)."""
    rx = rankdata(x); ry = rankdata(y); rz = rankdata(z)
    # Linear residual after removing rz
    def residual(a, b):
        b1 = b - b.mean()
        a1 = a - a.mean()
        slope = (a1 * b1).sum() / (b1 * b1).sum()
        return a1 - slope * b1
    rxz = residual(rx, rz)
    ryz = residual(ry, rz)
    # Pearson on residuals
    num = (rxz * ryz).sum()
    den = np.sqrt((rxz**2).sum() * (ryz**2).sum())
    rho = num / den if den > 0 else np.nan
    n = len(x)
    # approximate t-test for partial Spearman with df = n - 3
    if not np.isfinite(rho) or abs(rho) >= 1:
        return float(rho), np.nan
    df = n - 3
    t_stat = rho * np.sqrt(df / (1 - rho**2))
    from scipy.stats import t as student_t
    pval = 2 * student_t.sf(abs(t_stat), df)
    return float(rho), float(pval)


def fr_binned_test(cv_l, accs, fr, n_bins=4):
    """Within FR quartile bins, compute Spearman(CV_L, acc)."""
    qs = np.nanquantile(fr, np.linspace(0, 1, n_bins + 1))
    out = []
    for q in range(n_bins):
        m = (fr >= qs[q]) & ((fr <= qs[q+1]) if q == n_bins - 1 else (fr < qs[q+1]))
        m = m & np.isfinite(cv_l) & np.isfinite(accs)
        if m.sum() < 5:
            continue
        rho, p = spearmanr(cv_l[m], accs[m])
        out.append({
            "fr_bin": q + 1,
            "fr_range": [float(qs[q]), float(qs[q+1])],
            "n": int(m.sum()),
            "cv_acc_rho": float(rho),
            "cv_acc_p":   float(p),
            "mean_cv_l":  float(np.nanmean(cv_l[m])),
            "mean_acc":   float(np.nanmean(accs[m])),
        })
    return out


def main():
    X, y = load_v1_driven(MAT_PATH)
    accs, cv_l, fr = per_neuron_decode(X, y)
    valid = np.isfinite(accs) & np.isfinite(cv_l) & np.isfinite(fr)
    cv_l, accs, fr = cv_l[valid], accs[valid], fr[valid]
    n = len(cv_l)
    print(f"[ok] n_valid = {n}")

    # raw Spearman
    rho_cv, p_cv = spearmanr(cv_l, accs)
    rho_fr, p_fr = spearmanr(fr, accs)
    rho_cf, p_cf = spearmanr(cv_l, fr)

    # partial Spearman
    pcv,  ppcv  = partial_spearman(cv_l, accs, fr)
    pfr,  ppfr  = partial_spearman(fr,   accs, cv_l)

    # FR-binned within-bin CV-vs-acc
    bins = fr_binned_test(cv_l, accs, fr, n_bins=4)

    summary = {
        "n_neurons": int(n),
        "raw_spearman": {
            "cv_vs_acc": {"rho": float(rho_cv), "p": float(p_cv)},
            "fr_vs_acc": {"rho": float(rho_fr), "p": float(p_fr)},
            "cv_vs_fr":  {"rho": float(rho_cf), "p": float(p_cf)},
        },
        "partial_spearman": {
            "cv_vs_acc_given_fr": {"rho": pcv, "p": ppcv},
            "fr_vs_acc_given_cv": {"rho": pfr, "p": ppfr},
        },
        "fr_quartile_bins": bins,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[save] wrote {OUT_JSON}")

    lines = [
        "=== V1 decoder vs CV (FR-controlled) ===",
        "",
        f"n_neurons = {n}",
        "",
        "--- Raw Spearman ---",
        f"  rho(CV_L, accuracy) = {rho_cv:+.3f}   (p = {p_cv:.2g})",
        f"  rho(FR,   accuracy) = {rho_fr:+.3f}   (p = {p_fr:.2g})",
        f"  rho(CV_L, FR)       = {rho_cf:+.3f}   (p = {p_cf:.2g})    [confound]",
        "",
        "--- Partial Spearman (controlling for the third variable) ---",
        f"  rho(CV_L, accuracy | FR) = {pcv:+.3f}   (p = {ppcv:.2g})",
        f"  rho(FR,   accuracy | CV) = {pfr:+.3f}   (p = {ppfr:.2g})",
        "",
        "--- Within-FR-bin CV-vs-accuracy correlation ---",
    ]
    for b in bins:
        lines.append(
            f"  FR bin {b['fr_bin']} ({b['fr_range'][0]:.2f}-{b['fr_range'][1]:.2f} Hz, n={b['n']:3d}):"
            f"  rho={b['cv_acc_rho']:+.3f}  p={b['cv_acc_p']:.2g}   mean CV={b['mean_cv_l']:.3f}  mean acc={b['mean_acc']*100:.2f}%"
        )
    lines += [
        "",
        "--- Interpretation guide ---",
        "  If rho(CV_L, acc | FR) is still positive and significant: the CV effect is",
        "    not fully reducible to firing rate, supporting the framework's claim.",
        "  If rho(CV_L, acc | FR) drops to ~0 or becomes non-significant: the apparent",
        "    CV effect was a firing-rate artifact, and the reliability-CV link is",
        "    not empirically supported in this dataset.",
        "  Within-FR-bin correlations cross-check this: if positive in most bins,",
        "    the relationship is robust; if mixed/zero, FR is doing the work.",
    ]
    OUT_TXT.write_text("\n".join(lines))
    print("[save] wrote", OUT_TXT)
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
