"""
v1_decoder_vs_CV_multi.py
=========================
Multi-session decoder decomposition on the macaque V1 dataset.

Extends v1_decoder_vs_CV_partial.py from 1 session (105l001p16) to all 5
v1v2-1 sessions (564 V1 neurons total).  The analysis separates total lifetime
CV across all trials into stimulus-tuning dispersion and within-stimulus noise
dispersion because total CV contains the class structure used by the decoder.

Pipeline (per session)
----------------------
  1. Load .mat -> per-trial spike counts per V1 neuron (1.28-s driven window).
  2. For each V1 neuron, fit a 1D 8-class LDA (10-fold stratified CV) and
     record per-neuron held-out accuracy and per-neuron CV_L and mean FR.

Aggregate analysis
------------------
  3. Pool 564 (acc, CV_L, FR) triples across sessions; compute raw and partial
     Spearman correlations and within-FR-quartile correlations.
  4. Repeat the partial Spearman within each session separately (within-session
     no-pooling check).
  5. Optional: meta-analytic aggregation of the per-session partial rho via
     Fisher r-to-z (so a positive overall finding is not driven by one
     session).
  6. Save JSON/NPZ source data, a text report, and a summary PDF.

Output
------
  v1_decoder_vs_CV_multi.json
  v1_decoder_vs_CV_multi.npz
  v1_decoder_vs_CV_multi.txt
  v1_decoder_vs_CV_multi.pdf
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import spearmanr, rankdata, t as student_t, norm as norm_dist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict


HERE = Path(__file__).resolve().parent
# Keep headless figure generation reproducible in restricted reviewer/CI
# environments where user-level Matplotlib and fontconfig caches are read-only.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gibbsmanifold-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gibbsmanifold-cache")
SESSIONS = [
    "105l001p16.mat",
    "106r001p26.mat",
    "106r002p70.mat",
    "107l002p67.mat",
    "107l003p143.mat",
]

OUT_JSON = HERE / "v1_decoder_vs_CV_multi.json"
OUT_NPZ  = HERE / "v1_decoder_vs_CV_multi.npz"
OUT_TXT  = HERE / "v1_decoder_vs_CV_multi.txt"
OUT_PDF  = HERE / "v1_decoder_vs_CV_multi.pdf"

DRIVEN_BINS = 1280
N_CLASSES   = 8
RNG_SEED    = 20260502
KFOLDS      = 10


# --- I/O ------------------------------------------------------------------

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


# --- per-neuron decoder ---------------------------------------------------

def per_neuron_decode(X, y, k=KFOLDS, seed=RNG_SEED, label=""):
    n_trials, n_neurons = X.shape
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = np.full(n_neurons, np.nan)
    cv_l = np.full(n_neurons, np.nan)
    noise_cv = np.full(n_neurons, np.nan)
    tuning_cv = np.full(n_neurons, np.nan)
    fr   = np.full(n_neurons, np.nan)
    t0 = time.time()
    for j in range(n_neurons):
        x = X[:, j:j+1]
        m = x.mean()
        if m <= 0:
            continue
        cv_l[j] = x.std(ddof=1) / m
        class_means = np.array([x[y == cls].mean() for cls in np.unique(y)])
        within_vars = np.array([x[y == cls].var(ddof=1) for cls in np.unique(y)])
        noise_cv[j] = np.sqrt(np.nanmean(within_vars)) / m
        tuning_cv[j] = class_means.std(ddof=1) / max(class_means.mean(), 1e-12)
        fr[j]   = m / 1.28
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Use a deterministic single-process path so the analysis also
                # runs in restricted review/CI environments without joblib
                # worker or shared-memory support.
                y_pred = cross_val_predict(clf, x, y, cv=skf, n_jobs=1)
                accs[j] = (y_pred == y).mean()
            except Exception:
                pass
        if (j + 1) % 25 == 0:
            sys.stdout.write(f"\r    [{label}] neuron {j+1}/{n_neurons}, elapsed {time.time()-t0:.1f}s")
            sys.stdout.flush()
    print()
    return accs, cv_l, noise_cv, tuning_cv, fr


# --- statistics -----------------------------------------------------------

def partial_spearman(x, y, z):
    """Partial Spearman rho(x,y | z) via rank-residualization."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = np.asarray(x)[valid], np.asarray(y)[valid], np.asarray(z)[valid]
    if len(x) < 4:
        return float("nan"), float("nan")
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def residual(a, b):
        b1 = b - b.mean()
        a1 = a - a.mean()
        denom = (b1 * b1).sum()
        if denom == 0:
            return a1
        slope = (a1 * b1).sum() / denom
        return a1 - slope * b1
    rxz = residual(rx, rz)
    ryz = residual(ry, rz)
    num = (rxz * ryz).sum()
    den = np.sqrt((rxz**2).sum() * (ryz**2).sum())
    rho = float(num / den) if den > 0 else float("nan")
    n = len(x)
    df = n - 3
    if not np.isfinite(rho) or abs(rho) >= 1 or df <= 0:
        return rho, float("nan")
    t_stat = rho * np.sqrt(df / (1 - rho**2))
    pval = 2 * student_t.sf(abs(t_stat), df)
    return rho, float(pval)


def partial_spearman_multi(x, y, controls):
    """Partial Spearman rho from the inverse rank-correlation matrix."""
    variables = np.column_stack([x, y, *controls]).astype(float)
    variables = variables[np.all(np.isfinite(variables), axis=1)]
    if variables.shape[0] <= variables.shape[1]:
        return float("nan"), float("nan")
    corr = np.asarray(spearmanr(variables, axis=0).statistic, dtype=float)
    if corr.ndim != 2:
        return float("nan"), float("nan")
    precision = np.linalg.pinv(corr)
    rho = float(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))
    df = len(x) - len(controls) - 2
    t_stat = rho * np.sqrt(df / (1 - rho**2))
    p = 2 * student_t.sf(abs(t_stat), df)
    return rho, float(p)


def fr_binned_test(cv_l, accs, fr, n_bins=4):
    qs = np.nanquantile(fr, np.linspace(0, 1, n_bins + 1))
    out = []
    for q in range(n_bins):
        if q == n_bins - 1:
            m = (fr >= qs[q]) & (fr <= qs[q+1])
        else:
            m = (fr >= qs[q]) & (fr < qs[q+1])
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


def fisher_combine(rhos, ns):
    """Combine independent Spearman rho via Fisher z transform.
    Returns combined rho and a two-sided p-value on z=0 (i.e. test that the
    weighted-mean Fisher z is non-zero)."""
    rhos = np.asarray(rhos, dtype=float)
    ns   = np.asarray(ns, dtype=int)
    valid = np.isfinite(rhos) & (np.abs(rhos) < 1) & (ns > 3)
    rhos, ns = rhos[valid], ns[valid]
    if len(rhos) == 0:
        return float("nan"), float("nan")
    zs = 0.5 * np.log((1 + rhos) / (1 - rhos))   # Fisher z
    ws = ns - 3                                   # variance weights
    z_mean = (ws * zs).sum() / ws.sum()
    z_se = 1.0 / np.sqrt(ws.sum())
    z_stat = z_mean / z_se
    pval = 2 * norm_dist.sf(abs(z_stat))
    rho_combined = (np.exp(2 * z_mean) - 1) / (np.exp(2 * z_mean) + 1)
    return float(rho_combined), float(pval)


# --- main -----------------------------------------------------------------

def run_session(mat_name):
    p = HERE / mat_name
    print(f"\n[session] {mat_name}")
    X, y = load_v1_driven(p)
    print(f"  trials={X.shape[0]}, V1 neurons={X.shape[1]}, stims={sorted(set(y))}")
    accs, cv_l, noise_cv, tuning_cv, fr = per_neuron_decode(
        X, y, label=mat_name.split('.')[0]
    )
    valid = (
        np.isfinite(accs) & np.isfinite(cv_l) & np.isfinite(noise_cv)
        & np.isfinite(tuning_cv) & np.isfinite(fr)
    )
    cv_l, noise_cv, tuning_cv, accs, fr = (
        cv_l[valid], noise_cv[valid], tuning_cv[valid], accs[valid], fr[valid]
    )

    rho_cv,  p_cv  = spearmanr(cv_l, accs)
    rho_fr,  p_fr  = spearmanr(fr,   accs)
    rho_cf,  p_cf  = spearmanr(cv_l, fr)
    pcv,     ppcv  = partial_spearman(cv_l, accs, fr)
    pfr,     ppfr  = partial_spearman(fr,   accs, cv_l)
    ptotal_tuning, ptotal_tuning_p = partial_spearman_multi(
        cv_l, accs, [fr, tuning_cv]
    )
    pnoise, pnoise_p = partial_spearman(noise_cv, accs, fr)

    return {
        "session": mat_name.split('.')[0],
        "n_neurons": int(len(cv_l)),
        "median_acc": float(np.nanmedian(accs)),
        "median_cv_l": float(np.nanmedian(cv_l)),
        "median_fr":   float(np.nanmedian(fr)),
        "raw": {
            "cv_vs_acc": {"rho": float(rho_cv), "p": float(p_cv)},
            "fr_vs_acc": {"rho": float(rho_fr), "p": float(p_fr)},
            "cv_vs_fr":  {"rho": float(rho_cf), "p": float(p_cf)},
        },
        "partial": {
            "cv_acc_given_fr": {"rho": pcv, "p": ppcv},
            "fr_acc_given_cv": {"rho": pfr, "p": ppfr},
            "total_cv_acc_given_fr_and_tuning": {"rho": ptotal_tuning, "p": ptotal_tuning_p},
            "within_stim_noise_cv_acc_given_fr": {"rho": pnoise, "p": pnoise_p},
        },
        "_arrays": {
            "cv_l": cv_l.tolist(), "noise_cv": noise_cv.tolist(),
            "tuning_cv": tuning_cv.tolist(), "acc": accs.tolist(), "fr": fr.tolist(),
        },
    }


def main():
    per_session = []
    for s in SESSIONS:
        per_session.append(run_session(s))

    # --- pooled across sessions ---
    all_cv  = np.concatenate([np.asarray(r["_arrays"]["cv_l"]) for r in per_session])
    all_acc = np.concatenate([np.asarray(r["_arrays"]["acc"])  for r in per_session])
    all_fr  = np.concatenate([np.asarray(r["_arrays"]["fr"])   for r in per_session])
    all_noise_cv = np.concatenate([np.asarray(r["_arrays"]["noise_cv"]) for r in per_session])
    all_tuning_cv = np.concatenate([np.asarray(r["_arrays"]["tuning_cv"]) for r in per_session])
    print(f"\n[pooled] N = {len(all_cv)} neurons across {len(per_session)} sessions")

    rho_cv_p, p_cv_p = spearmanr(all_cv, all_acc)
    rho_fr_p, p_fr_p = spearmanr(all_fr, all_acc)
    rho_cf_p, p_cf_p = spearmanr(all_cv, all_fr)
    pcv_p, ppcv_p   = partial_spearman(all_cv, all_acc, all_fr)
    pfr_p, ppfr_p   = partial_spearman(all_fr, all_acc, all_cv)
    ptotal_tuning_p, pptotal_tuning_p = partial_spearman_multi(
        all_cv, all_acc, [all_fr, all_tuning_cv]
    )
    pnoise_p, ppnoise_p = partial_spearman(all_noise_cv, all_acc, all_fr)
    rho_tuning, p_tuning = spearmanr(all_tuning_cv, all_acc)
    bins_pooled = fr_binned_test(all_cv, all_acc, all_fr, n_bins=4)

    # --- meta-analytic aggregation of within-session partials ---
    rhos_within = [r["partial"]["cv_acc_given_fr"]["rho"] for r in per_session]
    ns_within   = [r["n_neurons"] for r in per_session]
    rho_meta, p_meta = fisher_combine(rhos_within, ns_within)

    summary = {
        "sessions": [{k: v for k, v in r.items() if k != "_arrays"}
                     for r in per_session],
        "n_pooled": int(len(all_cv)),
        "pooled": {
            "raw": {
                "cv_vs_acc": {"rho": float(rho_cv_p), "p": float(p_cv_p)},
                "fr_vs_acc": {"rho": float(rho_fr_p), "p": float(p_fr_p)},
                "cv_vs_fr":  {"rho": float(rho_cf_p), "p": float(p_cf_p)},
            },
            "partial": {
                "cv_acc_given_fr": {"rho": pcv_p, "p": ppcv_p},
                "fr_acc_given_cv": {"rho": pfr_p, "p": ppfr_p},
                "total_cv_acc_given_fr_and_tuning": {
                    "rho": ptotal_tuning_p, "p": pptotal_tuning_p
                },
                "within_stim_noise_cv_acc_given_fr": {
                    "rho": pnoise_p, "p": ppnoise_p
                },
                "tuning_cv_vs_acc_raw": {
                    "rho": float(rho_tuning), "p": float(p_tuning)
                },
            },
            "fr_quartile_bins": bins_pooled,
        },
        "meta_within_session_partial": {
            "rho_combined_fisher": rho_meta,
            "p_combined_fisher": p_meta,
            "per_session_rhos": rhos_within,
            "per_session_ns": ns_within,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] wrote {OUT_JSON}")
    session_labels = np.concatenate([
        np.repeat(r["session"], r["n_neurons"]) for r in per_session
    ])
    np.savez(
        OUT_NPZ,
        session=session_labels,
        total_cv_l=all_cv,
        tuning_cv=all_tuning_cv,
        within_stim_noise_cv=all_noise_cv,
        decoder_accuracy=all_acc,
        mean_firing_rate_hz=all_fr,
    )
    print(f"[save] wrote {OUT_NPZ}")

    # --- text report ---
    lines = [
        "=== V1 decoder vs CV: multi-session, FR-controlled ===",
        "",
        f"Sessions: {len(per_session)}, total V1 neurons: {len(all_cv)}",
        "",
        "--- Per-session partial Spearman rho(CV_L, accuracy | FR) ---",
    ]
    for r in per_session:
        pr = r["partial"]["cv_acc_given_fr"]
        lines.append(f"  {r['session']:>14s}  n={r['n_neurons']:3d}  "
                     f"raw rho(CV,acc)={r['raw']['cv_vs_acc']['rho']:+.3f}  "
                     f"partial rho={pr['rho']:+.3f}  p={pr['p']:.2g}")
    lines += [
        "",
        f"  Fisher-combined (meta) partial rho = {rho_meta:+.3f}  (p = {p_meta:.2g})",
        "",
        "--- Pooled across sessions (N = {} neurons) ---".format(len(all_cv)),
        f"  Raw     rho(CV_L, acc) = {rho_cv_p:+.3f}  (p = {p_cv_p:.2g})",
        f"  Raw     rho(FR,   acc) = {rho_fr_p:+.3f}  (p = {p_fr_p:.2g})",
        f"  Raw     rho(CV_L, FR)  = {rho_cf_p:+.3f}  (p = {p_cf_p:.2g})    [confound]",
        f"  Partial rho(CV_L, acc | FR) = {pcv_p:+.3f}  (p = {ppcv_p:.2g})",
        f"  Partial rho(FR,   acc | CV) = {pfr_p:+.3f}  (p = {ppfr_p:.2g})",
        f"  Partial rho(total CV_L, acc | FR, tuning CV) = {ptotal_tuning_p:+.3f}  (p = {pptotal_tuning_p:.2g})",
        f"  Partial rho(within-stim noise CV, acc | FR) = {pnoise_p:+.3f}  (p = {ppnoise_p:.2g})",
        f"  Raw rho(tuning CV, acc) = {rho_tuning:+.3f}  (p = {p_tuning:.2g})",
        "",
        "--- Pooled within-FR-quartile CV-vs-accuracy ---",
    ]
    for b in bins_pooled:
        lines.append(
            f"  FR Q{b['fr_bin']} ({b['fr_range'][0]:.2f}-{b['fr_range'][1]:.2f} Hz, n={b['n']:3d}):"
            f"  rho={b['cv_acc_rho']:+.3f}  p={b['cv_acc_p']:.2g}   "
            f"mean CV={b['mean_cv_l']:.3f}  mean acc={b['mean_acc']*100:.2f}%"
        )

    OUT_TXT.write_text("\n".join(lines))
    print(f"[save] wrote {OUT_TXT}")
    print()
    print("\n".join(lines))

    # --- figure ---
    make_decomposition_figure(
        all_cv, all_tuning_cv, all_noise_cv, all_acc, all_fr, summary
    )


def make_figure(per_session, all_cv, all_acc, all_fr, summary):
    """Three-panel causal-chain figure (constrained_layout for clean spacing):
        A  raw CV_L vs accuracy     (FR coloring hints at the confounder)
        B  CV_L vs FR               (the rho=-0.81 confounder revealed)
        C  4 within-FR-quartile mini scatters (FR control made concrete)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    chance = 1.0 / N_CLASSES
    acc_max = max(40.0, float(np.nanmax(all_acc * 100)) + 2.0)

    fig = plt.figure(figsize=(15.5, 4.6), constrained_layout=True)
    outer = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.0, 1.25], wspace=0.0)

    rho_raw = summary["pooled"]["raw"]["cv_vs_acc"]["rho"]
    p_raw   = summary["pooled"]["raw"]["cv_vs_acc"]["p"]
    rho_par = summary["pooled"]["partial"]["cv_acc_given_fr"]["rho"]
    p_par   = summary["pooled"]["partial"]["cv_acc_given_fr"]["p"]
    rho_cf  = summary["pooled"]["raw"]["cv_vs_fr"]["rho"]

    # ----- Panel A: pooled CV vs accuracy -----
    axA = fig.add_subplot(outer[0, 0])
    sc = axA.scatter(all_cv, all_acc * 100, c=all_fr, s=18,
                     cmap="viridis", edgecolor="white",
                     linewidth=0.3, alpha=0.85)
    axA.axhline(chance * 100, ls="--", color="grey", lw=0.8,
                label=f"chance ({chance*100:.1f}%)")
    axA.set_xlabel("Per-neuron lifetime $\\mathrm{CV}_L$")
    axA.set_ylabel("8-stim decoder accuracy (%)")
    axA.set_title(
        f"A.  Pooled raw relationship  ($N={len(all_cv)}$)\n"
        f"$\\rho_{{\\mathrm{{raw}}}}={rho_raw:+.2f}$ (p={p_raw:.0e}),   "
        f"$\\rho_{{\\,|\\,\\mathrm{{FR}}}}={rho_par:+.2f}$ (p={p_par:.0e})",
        fontsize=10
    )
    cbar = fig.colorbar(sc, ax=axA, pad=0.02, shrink=0.95)
    cbar.set_label("mean firing rate (Hz)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    axA.set_ylim(8, acc_max)
    axA.legend(fontsize=8, frameon=False, loc="upper right")

    # ----- Panel B: CV_L vs FR (the confounder) -----
    axB = fig.add_subplot(outer[0, 1])
    axB.scatter(all_fr, all_cv, color="#4a6fa5", s=18,
                edgecolor="white", linewidth=0.3, alpha=0.7)
    fr_log_bins = np.geomspace(max(all_fr.min(), 0.3), all_fr.max(), 12)
    bin_centers, bin_meds = [], []
    for k in range(len(fr_log_bins) - 1):
        m = (all_fr >= fr_log_bins[k]) & (all_fr < fr_log_bins[k+1])
        if m.sum() >= 5:
            bin_centers.append(np.sqrt(fr_log_bins[k] * fr_log_bins[k+1]))
            bin_meds.append(np.median(all_cv[m]))
    axB.plot(bin_centers, bin_meds, color="white", lw=3.0, zorder=4)
    axB.plot(bin_centers, bin_meds, color="black", lw=1.3, zorder=5,
             label="running median")
    axB.set_xscale("log")
    axB.set_xlabel("Mean firing rate (Hz, log scale)")
    axB.set_ylabel("Per-neuron lifetime $\\mathrm{CV}_L$")
    axB.set_title(
        f"B.  The confounder: $\\mathrm{{CV}}_L$ vs FR\n"
        f"$\\rho={rho_cf:+.2f}$ (p$<10^{{-100}}$); strong negative slope\n"
        f"is why raw $\\rho$(CV,acc) is suppressed in A",
        fontsize=10
    )
    axB.legend(fontsize=8, frameon=False, loc="upper right")

    # ----- Panel C: 2x2 within-FR-quartile mini scatters -----
    outerC = outer[0, 2]
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outerC,
                                    wspace=0.30, hspace=0.55)
    bins = summary["pooled"]["fr_quartile_bins"]
    fr_q = np.nanquantile(all_fr, np.linspace(0, 1, 5))
    quartile_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, 4))

    for q in range(4):
        if q == 3:
            mask = (all_fr >= fr_q[q]) & (all_fr <= fr_q[q+1])
        else:
            mask = (all_fr >= fr_q[q]) & (all_fr < fr_q[q+1])
        ax = fig.add_subplot(inner[q // 2, q % 2])
        b = bins[q]
        ax.scatter(all_cv[mask], all_acc[mask] * 100,
                   color=quartile_colors[q], s=10,
                   edgecolor="white", linewidth=0.25, alpha=0.85)
        ax.axhline(chance * 100, ls="--", color="grey", lw=0.5)
        # running-median trend within this FR quartile
        xs_in = all_cv[mask]
        ys_in = all_acc[mask] * 100
        if len(xs_in) >= 8:
            order = np.argsort(xs_in)
            xs_o, ys_o = xs_in[order], ys_in[order]
            k = max(3, len(xs_o) // 6)
            med_x, med_y = [], []
            for i0 in range(0, len(xs_o) - k + 1, max(1, k // 2)):
                med_x.append(np.median(xs_o[i0:i0+k]))
                med_y.append(np.median(ys_o[i0:i0+k]))
            ax.plot(med_x, med_y, color="black", lw=1.1, alpha=0.7)
        sig = ("***" if b["cv_acc_p"] < 0.001 else
               "**"  if b["cv_acc_p"] < 0.01  else
               "*"   if b["cv_acc_p"] < 0.05  else "n.s.")
        ax.set_title(
            f"FR Q{q+1}: {b['fr_range'][0]:.1f}-{b['fr_range'][1]:.1f} Hz   "
            f"$\\rho={b['cv_acc_rho']:+.2f}$ {sig} (n={b['n']})",
            fontsize=8.5
        )
        ax.tick_params(labelsize=7)
        ax.set_ylim(8, acc_max)
        if q in (2, 3):
            ax.set_xlabel("$\\mathrm{CV}_L$", fontsize=8.5)
        if q in (0, 2):
            ax.set_ylabel("accuracy (%)", fontsize=8.5)

    # Panel C header positioned over its own subplot region (after layout settles).
    fig.canvas.draw()
    bbox = outerC.get_position(fig)
    fig.text(bbox.x0 + bbox.width / 2, bbox.y1 + 0.02,
             "C.  FR control made concrete: within each FR quartile, $\\mathrm{CV}_L$ still predicts accuracy",
             ha="center", va="bottom", fontsize=10)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PDF}")


def make_decomposition_figure(
    all_cv, all_tuning_cv, all_noise_cv, all_acc, all_fr, summary
):
    """Show why total lifetime CV cannot be interpreted as noise variability."""
    import matplotlib.pyplot as plt

    pooled = summary["pooled"]
    total_raw = pooled["raw"]["cv_vs_acc"]
    total_fr = pooled["partial"]["cv_acc_given_fr"]
    tuning = pooled["partial"]["tuning_cv_vs_acc_raw"]
    noise = pooled["partial"]["within_stim_noise_cv_acc_given_fr"]
    total_tuning = pooled["partial"]["total_cv_acc_given_fr_and_tuning"]

    fig, axes = plt.subplots(
        1, 4, figsize=(15.2, 4.3), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.0, 0.9]},
    )

    sc = axes[0].scatter(
        all_cv, all_acc * 100, c=all_fr, s=18, cmap="viridis",
        edgecolor="white", linewidth=0.3, alpha=0.85,
    )
    axes[0].set_xlabel("Total lifetime $\\mathrm{CV}_L$")
    axes[0].set_ylabel("8-stim decoder accuracy (%)")
    axes[0].set_title(
        "A. Total CV includes tuning\n"
        f"$\\rho_{{raw}}={total_raw['rho']:+.2f}$; "
        f"$\\rho_{{|FR}}={total_fr['rho']:+.2f}$",
        fontsize=10,
    )
    cbar = fig.colorbar(sc, ax=axes[0], pad=0.02, shrink=0.92)
    cbar.set_label("mean firing rate (Hz)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    axes[1].scatter(
        all_tuning_cv, all_acc * 100, color="#c75b39", s=18,
        edgecolor="white", linewidth=0.3, alpha=0.75,
    )
    axes[1].set_xlabel("Tuning CV across stimulus means")
    axes[1].set_ylabel("8-stim decoder accuracy (%)")
    axes[1].set_title(
        "B. Tuning dispersion drives accuracy\n"
        f"$\\rho={tuning['rho']:+.2f}$ (p={tuning['p']:.0e})",
        fontsize=10,
    )

    axes[2].scatter(
        all_noise_cv, all_acc * 100, color="#4a6fa5", s=18,
        edgecolor="white", linewidth=0.3, alpha=0.75,
    )
    axes[2].set_xlabel("Within-stimulus noise CV")
    axes[2].set_ylabel("8-stim decoder accuracy (%)")
    axes[2].set_title(
        "C. Noise variability is detrimental\n"
        f"$\\rho_{{|FR}}={noise['rho']:+.2f}$ (p={noise['p']:.0e})",
        fontsize=10,
    )

    labels = ["Total CV\n| FR", "Total CV\n| FR,tuning", "Noise CV\n| FR"]
    vals = [total_fr["rho"], total_tuning["rho"], noise["rho"]]
    axes[3].bar(
        np.arange(3), vals, color=["#6a9f58", "#c75b39", "#4a6fa5"],
        edgecolor="black", linewidth=0.6,
    )
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set_xticks(np.arange(3), labels)
    axes[3].set_ylabel("Partial Spearman $\\rho$ with accuracy")
    axes[3].set_ylim(-0.7, 0.85)
    axes[3].set_title("D. Decomposition changes inference", fontsize=10)
    for i, value in enumerate(vals):
        axes[3].text(
            i, value + (0.04 if value >= 0 else -0.06), f"{value:+.2f}",
            ha="center", va="bottom" if value >= 0 else "top", fontsize=9,
        )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
