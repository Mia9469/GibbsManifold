"""
discriminative_maxent_test.py
=============================
Discriminative test of the Gibbs / maximum-entropy hypothesis on macaque V1.

Motivation
----------
The leave-one-session-out frontier control (Fig. S20) compared four functions
of the *information-vs-rate* curve and found that a rate-only linear model and a
sqrt curve out-predicted the one-parameter Gibbs reference.  Power-cost MaxEnt
families share a logarithmic entropy slope,

  For the maximum-entropy family p(r) wrt cost c(r)=r^alpha on [0, inf),
      p(r) propto exp(-beta r^alpha),
  the entropy obeys H(mu, alpha) = ln(mu) + h(alpha).

The offset h(alpha) remains shape dependent and is not erased algebraically by
a multiplicative R0.  In the present data it is weakly identified because R0,
the entropy reference level, and substantial rate-dependent noise entropy are
not independently known.  The frontier result is therefore an empirical limit,
not a structural theorem that alpha can never be identified.

The discriminative content of the Gibbs framework is therefore NOT the frontier
shape but the candidate-set claim that the *response distribution itself* is
better described by a mean-scaled power-cost maximum-entropy form
exp(-beta r^alpha)
than by a standard non-maximum-entropy competitor for skewed firing-rate
distributions (the lognormal), with a two-parameter gamma density added as a
stricter sensitivity analysis.

Two falsifiable predictions are checked:
  (T1) Held-out distribution fit: the maximum-entropy (Gibbs) form predicts
       held-out per-neuron response distributions better than the lognormal,
       neuron by neuron; a gamma comparator tests the boundary of that claim.
  (T2) Cross-moment agreement: the effective shape alpha estimated from
       the CV (2nd-moment) must agree with alpha estimated from full-likelihood
       (all moments).  A genuine generalized-gamma (Gibbs) law passes; a
       lognormal drawn population fails (it has no single consistent alpha).
       Negative control: simulated lognormal data is run through the identical
       pipeline and should show LOWER Gibbs-vs-lognormal win rate, confirming
       the test has discriminative power (is not a foregone conclusion).

Outputs
-------
  Fig5/discriminative_maxent_test.json
  updated/figures/Figure_discriminative.pdf / .png
"""
from __future__ import annotations

import argparse

import json
from pathlib import Path

import numpy as np
from scipy.special import digamma, gammaln, gamma, polygamma
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import spearmanr
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG2 = ROOT / "Fig2"
SESSIONS = [
    "105l001p16.mat",
    "106r001p26.mat",
    "106r002p70.mat",
    "107l002p67.mat",
    "107l003p143.mat",
]
DRIVEN_BINS = 1280
OUT_JSON = Path(__file__).resolve().parent / "discriminative_maxent_test.json"
OUT_FIG = ROOT / "updated" / "figures" / "Figure_discriminative.pdf"
OUT_PNG = OUT_FIG.with_suffix(".png")
SHARED_COUNT_JSON = Path(__file__).resolve().parent / "shared_shape_count_loso.json"
CROSSFIT_JSON = FIG2 / "v1_decoder_crossfit.json"
CROSSFIT_NPZ = FIG2 / "v1_decoder_crossfit.npz"

SEED = 20260627
N_SPLITS = 8           # train/test resamples per neuron
MIN_MEAN_COUNT = 5.0   # retain neurons with enough spikes for a shape test
MAX_ZERO_FRAC = 0.15   # avoid neurons dominated by exact zeros
ALPHA_LO, ALPHA_HI = 0.20, 8.0


# --------------------------------------------------------------------------
# Maximum-entropy family   p(r) propto exp(-beta r^alpha)  on [0, inf)
# (a generalized-gamma member; shape fixed by the effective exponent alpha)
# --------------------------------------------------------------------------

def cv_of_alpha(alpha: float) -> float:
    """Coefficient of variation of the maxent law -- depends ONLY on alpha."""
    g1, g2, g3 = gamma(1.0 / alpha), gamma(2.0 / alpha), gamma(3.0 / alpha)
    cv2 = g3 * g1 / (g2 * g2) - 1.0
    return float(np.sqrt(max(cv2, 0.0)))


def alpha_from_cv(cv: float) -> float:
    """Invert cv_of_alpha (monotone decreasing in alpha)."""
    cv = float(np.clip(cv, cv_of_alpha(ALPHA_HI) + 1e-6, cv_of_alpha(ALPHA_LO) - 1e-6))
    try:
        return float(brentq(lambda a: cv_of_alpha(a) - cv, ALPHA_LO, ALPHA_HI, xtol=1e-4))
    except ValueError:
        return float("nan")


def maxent_logpdf(r: np.ndarray, alpha: float, mu: float) -> np.ndarray:
    """log p(r) for the maxent law with mean fixed to mu (beta from mu, alpha)."""
    g1, g2 = gamma(1.0 / alpha), gamma(2.0 / alpha)
    beta = (g2 / (mu * g1)) ** alpha               # mean-matching constraint
    logZ = np.log(alpha) + (1.0 / alpha) * np.log(beta) - gammaln(1.0 / alpha)
    return logZ - beta * np.power(np.maximum(r, 1e-12), alpha)


def fit_maxent(train: np.ndarray) -> tuple[float, float]:
    """Mean-constrained MaxEnt fit: mu from training mean, alpha by 1-D likelihood search."""
    mu = float(train.mean())

    def nll(alpha):
        return -float(np.sum(maxent_logpdf(train, alpha, mu)))

    res = minimize_scalar(nll, bounds=(ALPHA_LO, ALPHA_HI), method="bounded")
    return float(res.x), mu


def lognormal_fit(train: np.ndarray) -> tuple[float, float]:
    """2-param MLE for lognormal on positive data."""
    logs = np.log(train)
    return float(logs.mean()), float(logs.std(ddof=1) + 1e-9)


def lognormal_logpdf(r: np.ndarray, m: float, s: float) -> np.ndarray:
    r = np.maximum(r, 1e-12)
    return -np.log(r * s * np.sqrt(2 * np.pi)) - (np.log(r) - m) ** 2 / (2 * s * s)


def gamma_fit(train: np.ndarray) -> tuple[float, float]:
    """2-param gamma MLE with loc fixed at 0, matched to the continuous-jittered counts."""
    x = np.maximum(train, 1e-12)
    mu = float(x.mean())
    s = float(np.log(mu) - np.log(x).mean())
    if s <= 1e-12:
        k = 200.0
    else:
        # Choi-Wette initialization followed by Newton updates for
        # log(k) - digamma(k) = log(mean(x)) - mean(log(x)).
        k = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
        k = float(np.clip(k, 1e-3, 200.0))
        for _ in range(20):
            f = np.log(k) - digamma(k) - s
            fp = 1.0 / k - polygamma(1, k)
            step = f / fp
            k_new = k - step
            if not np.isfinite(k_new) or k_new <= 0:
                k_new = 0.5 * k
            if abs(k_new - k) <= 1e-8 * max(1.0, k):
                k = k_new
                break
            k = float(np.clip(k_new, 1e-3, 200.0))
    theta = mu / k
    return k, theta


def gamma_logpdf(r: np.ndarray, k: float, theta: float) -> np.ndarray:
    r = np.maximum(r, 1e-12)
    return (k - 1.0) * np.log(r) - r / theta - k * np.log(theta) - gammaln(k)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_v1_counts(mat_path: Path, condition: str) -> list[np.ndarray]:
    """Per-neuron across-trial spike-count vectors for 'driven' or 'spontaneous'.

    Layout matches Fig2/v1_decoder_vs_CV_multi.py: rasters[trial, 0] is a
    (n_neurons x n_bins) raster.  Driven trials (stim!=0) span 1280 bins;
    spontaneous trials (stim==0) span 1500 bins and we take the first 1280
    (matched 1.28-s window, as in Methods).
    """
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    nd = m["neuralData"][0, 0]
    rasters = nd.spikeRasters
    stim = nd.stim.ravel().astype(int)
    rows = []
    for i in range(rasters.shape[0]):
        r = rasters[i, 0]
        if condition == "driven":
            if stim[i] == 0 or r.shape[1] < DRIVEN_BINS:
                continue
        else:  # spontaneous
            if stim[i] != 0 or r.shape[1] < DRIVEN_BINS:
                continue
        rows.append(np.asarray(r[:, :DRIVEN_BINS].sum(axis=1)).ravel().astype(float))
    if not rows:
        return []
    X = np.vstack(rows)                       # trials x neurons
    return [X[:, j] for j in range(X.shape[1])]


# --------------------------------------------------------------------------
# Per-neuron held-out comparison
# --------------------------------------------------------------------------

def passes_shape_qc(counts: np.ndarray) -> bool:
    c = np.asarray(counts, float)
    c = c[np.isfinite(c)]
    return bool(
        c.size >= 40
        and c.mean() >= MIN_MEAN_COUNT
        and np.mean(c == 0) <= MAX_ZERO_FRAC
    )


def held_out_compare(counts: np.ndarray, rng: np.random.Generator) -> dict | None:
    """Return per-neuron held-out log-likelihood contrasts and the two alpha estimates."""
    c = counts[np.isfinite(counts)]
    if not passes_shape_qc(c):
        return None
    # continuity jitter so the continuous models are well defined on integer counts
    cj_all = c + rng.uniform(0.0, 1.0, size=c.size)

    cv_emp = float(c.std(ddof=1) / c.mean())
    a_cv = alpha_from_cv(cv_emp)

    dll_ln = []       # test loglik per trial: Gibbs - lognormal
    dll_gamma = []    # test loglik per trial: Gibbs - gamma
    a_mle = []
    n = cj_all.size
    for _ in range(N_SPLITS):
        idx = rng.permutation(n)
        ntr = n // 2
        tr, te = cj_all[idx[:ntr]], cj_all[idx[ntr:]]
        if tr.min() <= 0 or te.size == 0:
            continue
        a_g, mu_g = fit_maxent(tr)
        m_ln, s_ln = lognormal_fit(tr)
        k_ga, th_ga = gamma_fit(tr)
        ll_g = float(np.mean(maxent_logpdf(te, a_g, mu_g)))
        ll_l = float(np.mean(lognormal_logpdf(te, m_ln, s_ln)))
        ll_ga = float(np.mean(gamma_logpdf(te, k_ga, th_ga)))
        dll_ln.append(ll_g - ll_l)
        dll_gamma.append(ll_g - ll_ga)
        a_mle.append(a_g)
    if not dll_ln:
        return None
    return {
        "cv": cv_emp,
        "alpha_from_cv": a_cv,
        "alpha_mle": float(np.nanmean(a_mle)),
        "dloglik_gibbs_minus_lognormal": float(np.mean(dll_ln)),
        "dloglik_gibbs_minus_gamma": float(np.mean(dll_gamma)),
        "mean_count": float(c.mean()),
    }


def run_population(
    per_neuron_counts: list[np.ndarray],
    rng,
    session: str | None = None,
) -> list[dict]:
    out = []
    for neuron_index, c in enumerate(per_neuron_counts):
        res = held_out_compare(np.asarray(c, float), rng)
        if res is not None:
            res["neuron_index"] = int(neuron_index)
            if session is not None:
                res["session"] = session.replace(".mat", "")
            out.append(res)
    return out


# --------------------------------------------------------------------------
# Negative control: simulated lognormal population through identical pipeline
# --------------------------------------------------------------------------

def simulate_lognormal_like(real: list[dict], rng) -> list[np.ndarray]:
    """Generate lognormal count vectors matched to the real CV/mean distribution."""
    sims = []
    for d in real:
        mu = d["mean_count"]
        cv = d["cv"]
        s2 = np.log(1 + cv * cv)
        m = np.log(mu) - 0.5 * s2
        x = rng.lognormal(mean=m, sigma=np.sqrt(s2), size=3000)
        sims.append(np.round(x))
    return sims


def simulate_gibbs_like(real: list[dict], rng) -> list[np.ndarray]:
    """Positive control: generalized-gamma (maxent) draws matched to real CV/mean."""
    sims = []
    for d in real:
        mu = d["mean_count"]
        a = d["alpha_from_cv"]
        if not np.isfinite(a):
            a = 1.0
        g1, g2 = gamma(1.0 / a), gamma(2.0 / a)
        beta = (g2 / (mu * g1)) ** a
        # sample r^alpha ~ Exponential(beta) over the base; r = (E/beta)^(1/alpha)
        e = rng.exponential(scale=1.0, size=3000)
        # density propto exp(-beta r^a): let u=beta r^a ~ Gamma(1/a,1)?  Use inverse-CDF via gamma:
        # r^a ~ Gamma(shape=1/a, scale=1/beta) gives p(r) propto r^{...}; instead sample directly:
        # exact: if Y~Gamma(1/a, 1) then r=(Y/beta)^(1/a) has pdf propto exp(-beta r^a).
        y = rng.gamma(shape=1.0 / a, scale=1.0, size=3000)
        r = (y / beta) ** (1.0 / a)
        sims.append(np.round(r))
    return sims


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def summarize(results: list[dict], label: str) -> dict:
    dll = np.array([r["dloglik_gibbs_minus_lognormal"] for r in results])
    dll_gamma = np.array([r["dloglik_gibbs_minus_gamma"] for r in results])
    a_cv = np.array([r["alpha_from_cv"] for r in results])
    a_ml = np.array([r["alpha_mle"] for r in results])
    ok = np.isfinite(a_cv) & np.isfinite(a_ml)
    # cross-moment agreement: correlation + median |alpha_cv - alpha_mle|
    if ok.sum() > 5:
        corr = float(np.corrcoef(a_cv[ok], a_ml[ok])[0, 1])
        gap = float(np.median(np.abs(a_cv[ok] - a_ml[ok])))
    else:
        corr, gap = float("nan"), float("nan")
    return {
        "label": label,
        "n_neurons": int(len(results)),
        "win_rate_gibbs": float(np.mean(dll > 0)),
        "win_rate_gibbs_vs_lognormal": float(np.mean(dll > 0)),
        "win_rate_gibbs_vs_gamma": float(np.mean(dll_gamma > 0)),
        "win_rate_gibbs_vs_both": float(np.mean((dll > 0) & (dll_gamma > 0))),
        "median_dloglik": float(np.median(dll)),
        "median_dloglik_gibbs_minus_lognormal": float(np.median(dll)),
        "median_dloglik_gibbs_minus_gamma": float(np.median(dll_gamma)),
        "mean_dloglik": float(np.mean(dll)),
        "mean_dloglik_gibbs_minus_lognormal": float(np.mean(dll)),
        "mean_dloglik_gibbs_minus_gamma": float(np.mean(dll_gamma)),
        "alpha_cv_median": float(np.nanmedian(a_cv)),
        "alpha_mle_median": float(np.nanmedian(a_ml)),
        "effective_shape_from_cv_median": float(np.nanmedian(a_cv)),
        "effective_shape_mle_median": float(np.nanmedian(a_ml)),
        "alpha_overid_corr": corr,
        "alpha_overid_median_gap": gap,
        "frac_sublinear_alpha_lt_1": float(np.mean(a_cv[np.isfinite(a_cv)] < 1.0)),
    }


def summarize_by_session(results: list[dict]) -> list[dict]:
    sessions = sorted({r.get("session") for r in results if r.get("session")})
    rows = []
    for session in sessions:
        subset = [r for r in results if r.get("session") == session]
        s = summarize(subset, session)
        rows.append({
            "session": session,
            "n_neurons": s["n_neurons"],
            "win_rate_vs_lognormal": s["win_rate_gibbs_vs_lognormal"],
            "median_dloglik_vs_lognormal": s["median_dloglik_gibbs_minus_lognormal"],
            "win_rate_vs_gamma": s["win_rate_gibbs_vs_gamma"],
            "median_dloglik_vs_gamma": s["median_dloglik_gibbs_minus_gamma"],
        })
    return rows


def run_condition(condition: str, rng) -> tuple[list[dict], dict]:
    real = []
    n_loaded = 0
    for s in SESSIONS:
        p = FIG2 / s
        if not p.exists():
            print(f"  [skip] {s} not found")
            continue
        per_neuron = load_v1_counts(p, condition)
        n_loaded += len(per_neuron)
        real.extend(run_population(per_neuron, rng, session=s))
    print(f"[{condition}] loaded {n_loaded} V1 neurons")
    print(f"[{condition}] retained {len(real)} neurons after shape-test QC")
    ln_sims = simulate_lognormal_like(real, rng)
    gibbs_sims = simulate_gibbs_like(real, rng)
    ln_res = run_population(ln_sims, rng)
    gibbs_res = run_population(gibbs_sims, rng)
    block = {
        "real_v1": summarize(real, f"real V1 ({condition})"),
        "real_v1_by_session": summarize_by_session(real),
        "neg_control_lognormal": summarize(ln_res, "simulated lognormal"),
        "pos_control_gibbs": summarize(gibbs_res, "simulated Gibbs"),
    }
    return real, block


def run_same_neuron_sensitivity() -> dict:
    """Restrict both conditions to neuron IDs passing QC in both states."""
    rng = np.random.default_rng(SEED + 1)
    spont_all, driven_all = [], []
    per_session = []
    for s in SESSIONS:
        p = FIG2 / s
        if not p.exists():
            continue
        spont = load_v1_counts(p, "spontaneous")
        driven = load_v1_counts(p, "driven")
        if len(spont) != len(driven):
            raise RuntimeError(f"condition neuron mismatch in {s}")
        sp_res, dr_res = [], []
        for neuron_index, (cs, cd) in enumerate(zip(spont, driven)):
            if not (passes_shape_qc(cs) and passes_shape_qc(cd)):
                continue
            rs = held_out_compare(cs, rng)
            rd = held_out_compare(cd, rng)
            if rs is None or rd is None:
                continue
            session = s.replace(".mat", "")
            for rec in (rs, rd):
                rec["session"] = session
                rec["neuron_index"] = int(neuron_index)
            sp_res.append(rs)
            dr_res.append(rd)
        spont_all.extend(sp_res)
        driven_all.extend(dr_res)
        per_session.append({
            "session": s.replace(".mat", ""),
            "n_neurons": len(sp_res),
            "spontaneous": summarize(sp_res, "spontaneous paired-QC"),
            "driven": summarize(dr_res, "driven paired-QC"),
        })
    return {
        "n_neurons": len(spont_all),
        "spontaneous": summarize(spont_all, "spontaneous paired-QC"),
        "driven": summarize(driven_all, "driven paired-QC"),
        "by_session": per_session,
    }


def main(make_figure: bool = True) -> None:
    rng = np.random.default_rng(SEED)
    driven, driven_block = run_condition("driven", rng)
    spont, spont_block = run_condition("spontaneous", rng)

    summary = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "driven": driven_block,
        "spontaneous": spont_block,
        "same_neuron_qc_sensitivity": run_same_neuron_sensitivity(),
        "note": (
            "win_rate_gibbs = fraction of neurons where the mean-scaled "
            "power-cost maximum-entropy form beats the lognormal on held-out "
            "log-likelihood; win_rate_gibbs_vs_gamma is the same comparison "
            "against a 2-parameter gamma competitor fit with loc fixed at zero. "
            "This is a candidate-set specification test, not an exhaustive "
            "model-selection claim. "
            "alpha_overid_corr describes whether the effective shape inferred from CV "
            "matches the one inferred from full likelihood (cross-moment "
            "agreement). Driven = stimulus-shaped; spontaneous = "
            "cost-constrained readiness regime (the framework's home turf)."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if make_figure:
        _make_figure(driven, spont, summary)
    print(f"Wrote {OUT_JSON}")
    if make_figure:
        print(f"Wrote {OUT_FIG}")


def _alpha_curve():
    aa = np.linspace(ALPHA_LO, 3.0, 200)
    return aa, np.array([cv_of_alpha(a) for a in aa])


def _load_decoder_data():
    """Cross-fitted per-neuron display arrays and session-level inference."""
    npz_path = CROSSFIT_NPZ
    json_path = CROSSFIT_JSON
    if not npz_path.exists() or not json_path.exists():
        return None, None
    npz = np.load(npz_path, allow_pickle=True)
    j = json.loads(json_path.read_text(encoding="utf-8"))
    return npz, j


def _load_shared_count_results():
    if not SHARED_COUNT_JSON.exists():
        return None
    return json.loads(SHARED_COUNT_JSON.read_text(encoding="utf-8"))


def _make_figure(driven, spont, summary):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, mos = plt.subplot_mosaic(
        [["A", "B", "C"], ["D", "E", "F"]], figsize=(12.4, 7.6)
    )
    axes = [mos["A"], mos["B"], mos["C"]]

    # A: held-out dloglik, spontaneous vs driven
    ax = axes[0]
    for res, col, lab in [(spont, "#4C78A8", "spontaneous"),
                          (driven, "#E15759", "driven")]:
        dll = np.array([r["dloglik_gibbs_minus_lognormal"] for r in res])
        ax.hist(dll, bins=40, color=col, alpha=0.55, edgecolor="none",
                label=f"{lab} (median {np.median(dll):+.3f})")
    ax.axvline(0, color="0.2", lw=1.0)
    ax.set_xlabel(r"held-out $\Delta\log L$ (power-cost MaxEnt $-$ lognormal)")
    ax.set_ylabel("neurons")
    ax.set_title("A  Distribution-level specification test")
    ax.legend(frameon=False, fontsize=7.2)

    # B: CV(alpha) backbone with both conditions' inferred alpha placed on it
    ax = axes[1]
    aa, cc = _alpha_curve()
    ax.plot(aa, cc, color="0.25", lw=1.6)
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    xmax = 3.0
    for res, col, lab in [(spont, "#4C78A8", "spontaneous"),
                          (driven, "#E15759", "driven")]:
        a_cv = np.array([r["alpha_from_cv"] for r in res])
        med_a = np.nanmedian(a_cv)
        x_disp = min(med_a, xmax - 0.05)          # clamp into view
        suffix = r"$\alpha\!\approx\!$" + f"{med_a:.2f}"
        if med_a > xmax:
            suffix += " (off-axis, concentrated)"
        ax.scatter([x_disp], [cv_of_alpha(med_a)], s=70, color=col,
                   edgecolor="black", lw=0.6, zorder=4,
                   marker=(">" if med_a > xmax else "o"),
                   label=f"{lab}: " + suffix)
    ax.annotate("exponential\n(linear cost)", xy=(1.0, 1.0), xytext=(1.5, 1.35),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel(r"effective shape $\alpha$")
    ax.set_ylabel(r"$\mathrm{CV}(\alpha)$ of maxent law")
    ax.set_xlim(ALPHA_LO, 3.0)
    ax.set_title("B  Effective shape parameter")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")

    # C: native-count, shared-shape LOSO comparison against the full candidate set
    ax = axes[2]
    shared = _load_shared_count_results()
    if shared is not None:
        x = np.arange(2)
        width = 0.22
        alternatives = [
            ("lognormal", "#B07AA1", "lognormal"),
            ("gamma", "#F28E2B", "gamma"),
            ("negative_binomial", "#4E79A7", "negative binomial"),
        ]
        for j, (alt, color, label) in enumerate(alternatives):
            xpos = x + (j - 1) * width
            conditions = []
            for condition in ("spontaneous", "driven"):
                folds = shared["conditions"][condition]["folds"]
                conditions.append(np.asarray([
                    fold["power_cost_contrasts"][alt]["mean_delta_elpd_per_trial"]
                    for fold in folds
                ]))
            means = [float(np.mean(values)) for values in conditions]
            ax.bar(xpos, means, width=width, color=color, alpha=0.72, label=label, zorder=2)
            for xi, values in zip(xpos, conditions):
                ax.scatter(np.full(values.size, xi), values, color=color, edgecolor="white",
                           linewidth=0.35, s=18, zorder=4)
        ax.axhline(0, color="0.25", lw=1.0)
        ax.set_xticks(x, ["spontaneous", "driven"])
        ax.set_ylabel(r"$\Delta$ELPD/trial: power-cost $-$ comparator")
        ax.set_title("C  Native-count shared-shape LOSO")
        ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    else:
        ax.text(0.5, 0.5, "shared-shape count results unavailable", ha="center", va="center")
        ax.set_axis_off()

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    # ---- bottom row: independently cross-fitted decoder evidence ----
    npz, jd = _load_decoder_data()
    axD, axE, axF = mos["D"], mos["E"], mos["F"]
    if npz is not None:
        prefixes = [session["session"] for session in jd["per_session"]]
        acc = np.concatenate([np.asarray(npz[f"{prefix}_accuracy"], float) for prefix in prefixes])
        cvl = np.concatenate([np.asarray(npz[f"{prefix}_total_cv"], float) for prefix in prefixes])
        fr = np.concatenate([np.asarray(npz[f"{prefix}_fr"], float) for prefix in prefixes])
        good = np.isfinite(acc) & np.isfinite(cvl) & np.isfinite(fr)
        acc, cvl, fr = acc[good], cvl[good], fr[good]
        rho_raw = float(spearmanr(cvl, acc).statistic)
        rho_par = jd["session_level"]["total_cv_accuracy_given_fr"]["fisher_mean"]

        # D: accuracy vs total CV_L, coloured by firing rate (shows FR confound)
        sc = axD.scatter(cvl, acc, c=np.log10(fr + 1e-3), cmap="viridis",
                         s=13, alpha=0.7, edgecolor="none")
        cb = fig.colorbar(sc, ax=axD, fraction=0.046, pad=0.03)
        cb.set_label(r"$\log_{10}$ firing rate", fontsize=7.5)
        axD.axhline(1.0 / 8, color="0.4", lw=0.8, ls=":")
        axD.text(0.02, 1.0 / 8 + 0.01, "chance", fontsize=6.8,
                 transform=axD.get_yaxis_transform())
        axD.set_xlabel(r"training-half total $\mathrm{CV}_L$")
        axD.set_ylabel("independent-half decoder accuracy")
        axD.set_title("D  Cross-fitted decoding")
        axD.text(0.03, 0.97,
                 f"display raw $\\rho={rho_raw:+.2f}$\nsession mean $\\rho_{{|\\mathrm{{FR}}}}={rho_par:+.2f}$",
                 transform=axD.transAxes, va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6))

        # E: one point per session; repeated splits do not add degrees of freedom.
        rows = [
            (r"total CV$\mid$FR", "total_cv_accuracy_given_fr"),
            (r"tuning CV$\mid$FR", "tuning_cv_accuracy_given_fr"),
            (r"noise CV$\mid$FR", "noise_cv_accuracy_given_fr"),
            (r"tuning CV$\mid$FR,MD,OSI,SNR", "tuning_incremental_given_fr_modulation_osi_snr"),
        ]
        labels = [row[0] for row in rows]
        ypos = np.arange(len(rows))[::-1]
        for yp, (_, key) in zip(ypos, rows):
            vals = np.asarray(jd["session_level"][key]["per_session"], float)
            axE.plot([vals.min(), vals.max()], [yp, yp], color="#A9BCD0", lw=2, zorder=1)
            axE.scatter(vals, np.full(vals.size, yp), color="#4C78A8", s=22, zorder=3)
            fisher_mean = float(jd["session_level"][key]["fisher_mean"])
            axE.scatter(fisher_mean, yp, color="#E15759", marker="D", s=30, zorder=4)
        axE.axvline(0, color="0.2", lw=1.0)
        axE.set_yticks(ypos, labels, fontsize=7.6)
        axE.set_xlabel(r"cross-fitted session Spearman $\rho$")
        axE.set_xlim(-0.55, 1.0)
        axE.set_title("E  Tuning vs conventional selectivity")
        # F: the CV--firing-rate confound that necessitates the FR control in D
        rho_cvfr = float(spearmanr(cvl, fr).statistic)
        axF.scatter(fr, cvl, s=12, alpha=0.5, color="#6B6B6B", edgecolor="none")
        axF.set_xscale("log")
        axF.set_xlabel("mean firing rate [Hz]")
        axF.set_ylabel(r"total $\mathrm{CV}_L$")
        axF.set_title("F  CV--rate confound")
        axF.text(0.03, 0.05, f"$\\rho={rho_cvfr:+.2f}$",
                 transform=axF.transAxes, va="bottom", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6))
        for a in (axD, axE, axF):
            a.spines[["top", "right"]].set_visible(False)
    else:
        axF.axis("off")

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-figure", action="store_true",
        help="recompute JSON summaries without overwriting manuscript figures",
    )
    args = parser.parse_args()
    main(make_figure=not args.no_figure)
