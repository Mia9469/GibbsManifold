"""noisy_channel_gibbs.py

Generates Supplementary Fig. S17 (noise-penalized entropy surrogate) and prints
the diagnostics quoted in Supp. Sect. "Noise-penalized differential-entropy
surrogate".

For a fixed response penalty g(r), maximize J_g = h - E[g].
This is not optimization of channel mutual information. The optimizer is
    q*(r)  proportional to  exp[ -g(r) - beta c(r) ]   on [0, r_max].
We study the Gaussian-rate noise family with local variance
    sigma^2(r) = sigma0^2 + alpha r        (baseline + Fano term),
    g(r) = (1/2) log( 2 pi e ( sigma0^2 + alpha r ) ),
parameterised by xi = sigma0^2 / (alpha * <r>_ref), which interpolates
between logarithmic rate penalties and baseline-dominated penalties.

Four panels:
  A  q*(r) for xi in {0.01, 0.3, 3, 30} at fixed beta (shape change)
  B  CV*(C) for the four regimes, showing where CV exceeds the entropy-only
     bounded-linear interval or ceases to be one-to-one in the budget
  C  J_g / C vs CV in beta order, including negative surrogate yields
  D  V1 data: H and I vs log(C - c0) with fitted semi-log slopes; the slope
     ratio b_I/b_H is compared descriptively with the empirical mean
     information fraction (1 - rho_bar).

Outputs:
  ../outputs/final_text_review_20260907/v1_corrected/FigureS17.pdf (and .png)
  prints the numbers used in the supp caption / main text

Usage:
  python noisy_channel_gibbs.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad, quad_vec
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- calibrated affine cost proxy (adapted from Yu et al. 2018) -----------
C0 = 2.4          # proxy baseline
C1 = 0.37         # proxy slope
LAM_MAX = 100.0   # Hz, bounded support

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from Fig5.v1_original_rates import DEFAULT_OUTPUT_DIR, RATE_PROVENANCE, load_evoked_sessions

CV_LO, CV_HI = 1 / np.sqrt(3), 1.0
CONFIGS = [
    ("rate-dominated (xi=0.01)",   0.01),
    ("mixed penalty (xi=0.3)",       0.30),
    ("baseline-rich (xi=3)",     3.0),
    ("baseline-dominated (xi=30)",   30.0),
]


def gibbs_sweep(xi, beta_grid, lam, dlam):
    """Return dict of CV, ICE_H, ICE_I, C, I, H, lam_mean over the beta grid."""
    alpha = 1.0
    sigma0sq = xi * alpha * 10.0  # reference <lam> = 10 Hz
    var_r = sigma0sq + alpha * lam
    g = 0.5 * np.log(2 * np.pi * np.e * var_r)

    out = {k: [] for k in ("beta", "CV", "ICE_H", "ICE_I", "C", "I", "H", "lam")}
    for beta in beta_grid:
        def integrand(r):
            relative_penalty = 0.5 * np.log1p(r / sigma0sq)
            weight = np.exp(-relative_penalty - beta * C1 * r)
            return weight * np.array([1.0, r, r * r, relative_penalty])

        integrals, error = quad_vec(integrand, 0.0, LAM_MAX,
                                    epsabs=1e-11, epsrel=1e-11)
        z = integrals[0]
        mlam = integrals[1] / z
        vlam = integrals[2] / z - mlam * mlam
        if z <= 0 or vlam <= 0 or not np.isfinite(integrals).all():
            raise ValueError("Invalid continuum surrogate moments")
        cv = np.sqrt(vlam) / max(mlam, 1e-12)
        cost = C0 + C1 * mlam
        relative_gmean = integrals[3] / z
        H = np.log(z) + relative_gmean + beta * C1 * mlam
        gmean = 0.5 * np.log(2 * np.pi * np.e * sigma0sq) + relative_gmean
        I_star = H - gmean

        out["beta"].append(beta)
        out["CV"].append(cv)
        out["ICE_H"].append(H / cost)
        out["ICE_I"].append(I_star / cost)
        out["C"].append(cost)
        out["I"].append(I_star)
        out["H"].append(H)
        out["lam"].append(mlam)
    return {k: np.asarray(v) for k, v in out.items()}


def example_density(xi, beta, lam, dlam):
    alpha = 1.0
    sigma0sq = xi * alpha * 10.0
    g = 0.5 * np.log(2 * np.pi * np.e * (sigma0sq + alpha * lam))
    p = np.exp(-0.5 * np.log1p(lam / sigma0sq) - beta * C1 * lam)
    z = quad(lambda r: np.exp(-0.5 * np.log1p(r / sigma0sq)
                              - beta * C1 * r),
             0.0, LAM_MAX, epsabs=1e-11, epsrel=1e-11)[0]
    return p / z


def load_v1():
    sessions = load_evoked_sessions()
    hr, mi, C, rho = [np.concatenate([row[key] for row in sessions])
                      for key in ("hr", "mi", "cost", "rho")]
    keep = (
        np.isfinite(hr) & np.isfinite(mi) & np.isfinite(C) & np.isfinite(rho)
        & (C > C0 + 0.1) & (mi > 0) & (hr > 0)
    )
    return hr[keep], mi[keep], C[keep], rho[keep]


def main(output_dir: Path = DEFAULT_OUTPUT_DIR):
    lam = np.linspace(1e-4, LAM_MAX, 4000)
    dlam = lam[1] - lam[0]
    beta_grid = np.logspace(-3, 1.5, 300)

    sweeps = [(name, xi, gibbs_sweep(xi, beta_grid, lam, dlam))
              for name, xi in CONFIGS]

    # --- trajectory diagnostics in beta order --------------------------
    print("=== Noisy-channel trajectory diagnostics (mean rate 0.5--60 Hz) ===")
    for name, xi, s in sweeps:
        m = (s["lam"] >= 0.5) & (s["lam"] <= 60)
        cv = s["CV"][m]
        dcv = np.diff(cv)
        turns = int(np.sum(np.sign(dcv[1:]) != np.sign(dcv[:-1])))
        frac_outside = float(np.mean((cv < CV_LO) | (cv > CV_HI)))
        print(
            f"  {name:26s} CV=[{cv.min():.3f}, {cv.max():.3f}], "
            f"outside entropy-only interval={100*frac_outside:4.1f}%, "
            f"CV direction changes={turns}"
        )

    # --- V1 frontier slopes -------------------------------------------
    hr, mi, C, rho = load_v1()
    lnC = np.log(C - C0)
    bH = np.polyfit(lnC, hr, 1)[0]
    bI = np.polyfit(lnC, mi, 1)[0]
    rho_bar = rho.mean()
    print("\n=== V1 frontier (semi-log Y = a + b ln(C - C0)) ===")
    print(f"  n = {len(hr)}, rho_bar = {rho_bar:.3f}, IQR width = "
          f"{np.percentile(rho,75)-np.percentile(rho,25):.3f}")
    print(f"  slope_H = {bH:.2f}, slope_I = {bI:.2f}, ratio = {bI/bH:.3f}")
    print(f"  empirical mean information fraction 1 - rho_bar = {1-rho_bar:.3f}")
    print("  This agreement is descriptive and does not identify the noise mechanism.")

    # --- figure --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(CONFIGS)))

    # A: example densities at fixed beta
    axA = axes[0, 0]
    beta_ex = 0.05
    for (name, xi, _), c in zip(sweeps, cmap):
        p = example_density(xi, beta_ex, lam, dlam)
        axA.plot(lam, p, color=c, lw=2, label=name)
    axA.set_xlabel("firing rate $r$ [Hz]")
    axA.set_ylabel(r"$q^*(r)$ at $\beta=0.05$")
    axA.set_title(r"A  Surrogate optimizer $q^*\propto e^{-g(r)-\beta c(r)}$")
    axA.set_xlim(0, 60)
    axA.legend(fontsize=8, loc="upper right")
    axA.grid(alpha=0.3)

    # B: CV vs C
    axB = axes[0, 1]
    for (name, xi, s), c in zip(sweeps, cmap):
        m = (s["lam"] >= 0.5) & (s["lam"] <= 60)
        axB.plot(s["C"][m], s["CV"][m], color=c, lw=2, label=name)
    axB.axhline(CV_LO, color="k", ls="--", lw=0.7)
    axB.axhline(CV_HI, color="k", ls="--", lw=0.7)
    axB.axhline(np.sqrt(2), color="r", ls=":", lw=1.2)
    axB.text(axB.get_xlim()[1]*0.98, np.sqrt(2)+0.02, r"$\sqrt{2}$ (case B, unbounded)",
             color="r", ha="right", va="bottom", fontsize=8)
    axB.text(axB.get_xlim()[1]*0.98, CV_LO-0.04, r"$1/\sqrt{3}$", ha="right", va="top", fontsize=8)
    axB.set_xlabel(r"Affine model cost $C$")
    axB.set_ylabel(r"$\mathrm{CV}^*$")
    axB.set_title(r"B  Penalty changes the optimal dispersion")
    axB.grid(alpha=0.3)

    # C: ICE_I vs CV, preserving beta order so folds remain visible.
    axC = axes[1, 0]
    axC.axvspan(CV_LO, CV_HI, color="0.9", zorder=0)
    # ── auto-range from all four curves ──
    cv_all, ice_all = [], []
    for (name, xi, s), c in zip(sweeps, cmap):
        m = (s["lam"] >= 0.5) & (s["lam"] <= 60)
        cv = s["CV"][m]; iI = s["ICE_I"][m]
        axC.plot(cv, iI, color=c, lw=2, label=name)
        cv_all.append(cv); ice_all.append(iI)
    cv_all = np.concatenate(cv_all)
    ice_all = np.concatenate(ice_all)
    x_lo = max(0.3, np.nanmin(cv_all) - 0.05)
    x_hi = max(1.5, np.nanmax(cv_all) + 0.05)
    y_hi = np.nanmax(ice_all) * 1.10
    axC.set_xlim(x_lo, x_hi)
    axC.set_ylim(min(0, float(np.min(ice_all)) * 1.10), y_hi)
    axC.set_xlabel(r"$\mathrm{CV}^*$")
    axC.set_ylabel(r"Surrogate yield $J_g/C$")
    axC.set_title(r"C  Surrogate-yield trajectories can fold")
    axC.legend(fontsize=8, loc="best")
    axC.grid(alpha=0.3)

    # D: V1 frontier slopes
    axD = axes[1, 1]
    axD.scatter(lnC, hr, s=8, color="tab:blue", alpha=0.35, label=r"Fitted $H(W)$")
    axD.scatter(lnC, mi, s=8, color="tab:orange", alpha=0.35, label=r"Fitted $I((S,J);W)$")
    xx = np.linspace(lnC.min(), lnC.max(), 50)
    aH = np.polyfit(lnC, hr, 1); aI = np.polyfit(lnC, mi, 1)
    axD.plot(xx, np.polyval(aH, xx), "b-", lw=2)
    axD.plot(xx, np.polyval(aI, xx), color="tab:orange", lw=2)
    axD.set_xlabel(r"$\ln(\widetilde C - C_0)$")
    axD.set_ylabel("rate [bits/s]")
    axD.set_title("D  V1 descriptive scaling: $b_I/b_H$ = %.2f; $1-\\bar\\rho$ = %.2f"
                  % (bI/bH, 1-rho_bar))
    axD.legend(fontsize=8, loc="upper left")
    axD.grid(alpha=0.3)

    fig.suptitle("Noise-penalized differential entropy; V1 estimator outputs are bias-sensitive",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / "FigureS17.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "v1_rate_provenance": RATE_PROVENANCE,
        "n_neurons": int(hr.size),
        "H_slope_bits_per_s": float(bH),
        "I_slope_bits_per_s": float(bI),
        "slope_ratio_I_over_H": float(bI / bH),
        "mean_information_fraction": float(1 - rho_bar),
    }
    (output_dir / "figureS17_v1_slopes.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    main(parser.parse_args().output_dir)
