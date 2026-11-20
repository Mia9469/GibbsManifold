"""Generate missing supplementary figures S12--S14 from cached results.

This script avoids MATLAB by reading the existing .mat intermediates:

* Fig. S12 from unified_v1_Hrho_ICE.mat + v1_CVL_perN.mat
* Fig. S13 from ../FigS4/alpha_sweep_results.mat
* Fig. S14 from variance_decomp_CVP.mat

Outputs are written both to this folder (for local inspection) and to
../updated/figures/ for manuscript compilation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
FIG2 = ROOT / "Fig2"
FIGS4 = ROOT / "FigS4"
OUT_MANUSCRIPT = ROOT / "updated" / "figures"


COL_SPON = np.array([0.30, 0.60, 0.90])
COL_EVK = np.array([0.85, 0.33, 0.10])

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
    }
)


def save_both(fig: plt.Figure, stem: str) -> None:
    """Save a figure locally and into updated/figures."""
    for out_dir in (FIG2, OUT_MANUSCRIPT):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    print(f"[save] {stem}.pdf/.png -> {FIG2} and {OUT_MANUSCRIPT}")


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Spearman partial correlation via rank residualization on z."""
    rx = np.asarray(rankdata(x), dtype=np.float64)
    ry = np.asarray(rankdata(y), dtype=np.float64)
    rz = np.asarray(rankdata(z), dtype=np.float64)
    rz_centered = rz - rz.mean()
    denom = np.dot(rz_centered, rz_centered)
    if denom <= 0:
        raise ValueError("partial Spearman covariate has zero rank variance")
    ex = (rx - rx.mean()) - np.dot(rz_centered, rx - rx.mean()) / denom * rz_centered
    ey = (ry - ry.mean()) - np.dot(rz_centered, ry - ry.mean()) / denom * rz_centered
    return spearmanr(ex, ey)


def load_v1_driven():
    """Load 5-session V1 driven H, MI, C and FR in session order."""
    S = loadmat(FIG2 / "unified_v1_Hrho_ICE.mat", squeeze_me=True, struct_as_record=False)
    sessions = np.ravel(S["results"].v1v2)
    hr, mi, cost, fr = [], [], [], []
    for sd in sessions:
        hr.append(np.ravel(sd.hr).astype(float))
        mi.append(np.ravel(sd.mi).astype(float))
        cost.append(np.ravel(sd.C).astype(float))
        fr.append(np.ravel(sd.FR).astype(float))
    hr = np.concatenate(hr)
    mi = np.concatenate(mi)
    cost = np.concatenate(cost)
    fr = np.concatenate(fr)
    keep = np.isfinite(hr) & np.isfinite(mi) & np.isfinite(cost) & (cost > 0)
    return hr[keep], mi[keep], cost[keep], fr[keep], keep


def make_s12() -> None:
    hr, mi, cost, fr, keep = load_v1_driven()
    L = loadmat(FIG2 / "v1_CVL_perN.mat", squeeze_me=True, struct_as_record=False)
    cv = np.ravel(L["cvL_drv"]).astype(float)
    if cv.size != keep.size:
        raise ValueError(f"CV_L length {cv.size} does not match driven length {keep.size}")
    cv = cv[keep]

    m = np.isfinite(cv) & (cv > 0) & np.isfinite(fr) & (fr > 0)
    cv = cv[m]
    fr = fr[m]
    ice_h = hr[m] / cost[m]
    ice_i = mi[m] / cost[m]

    rho_h, _ = spearmanr(cv, ice_h)
    rho_i, _ = spearmanr(cv, ice_i)
    pr_h, _ = partial_spearman(cv, ice_h, fr)
    pr_i, _ = partial_spearman(cv, ice_i, fr)

    with (FIG2 / "v1_ice_cvl_session_stats.json").open() as f:
        session_stats = json.load(f)
    session_h = session_stats["session_level_H_over_cost"]
    session_i = session_stats["session_level_I_over_cost"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharex=True)
    panels = [
        (axes[0], ice_h, session_h, r"$H / \widetilde{C}$", "A"),
        (axes[1], ice_i, session_i, r"$\mathrm{MI} / \widetilde{C}$", "B"),
    ]
    for ax, y, stats, ylabel, letter in panels:
        sc = ax.scatter(cv, y, c=fr, s=22, cmap="viridis", alpha=0.75,
                        edgecolor="none")
        # Running median for visual trend.
        order = np.argsort(cv)
        bins = np.array_split(order, 16)
        bx = np.array([np.median(cv[b]) for b in bins if len(b)])
        by = np.array([np.median(y[b]) for b in bins if len(b)])
        ax.plot(bx, by, color="black", lw=2.0, label="running median")
        ax.set_xlabel(r"$\mathrm{CV}_L$ (evoked, per neuron)")
        ax.set_ylabel(ylabel)
        direction_count = (
            stats["sessions_with_positive_rho"]
            if stats["fisher_mean_rho"] > 0
            else stats["n_sessions"] - stats["sessions_with_positive_rho"]
        )
        ax.text(
            0.03,
            0.96,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            0.97,
            0.96,
            rf"FR-controlled session mean $\rho={stats['fisher_mean_rho']:+.2f}$ ({direction_count}/5)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.07, bottom=0.16, right=0.88, top=0.95, wspace=0.30)
    cax = fig.add_axes([0.90, 0.19, 0.015, 0.69])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$\bar r$ (Hz)")
    save_both(fig, "FigureS12")
    plt.close(fig)
    print(
        "[S12] raw H/I rho:",
        f"{rho_h:+.3f}/{rho_i:+.3f}; partial H/I rho: {pr_h:+.3f}/{pr_i:+.3f}",
    )


def make_s13() -> None:
    S = loadmat(FIGS4 / "alpha_sweep_results.mat", squeeze_me=True, struct_as_record=False)
    alpha_list = np.ravel(S["alpha_list"]).astype(float)
    beta_grid = np.ravel(S["beta_grid"]).astype(float)
    r = np.ravel(S["r"]).astype(float)
    A_cost = float(S["A_cost"])
    CV_arr = np.asarray(S["CV_arr"], dtype=float)
    ICE_arr = np.asarray(S["ICE_arr"], dtype=float)
    cv_v1_evk = float(S["cv_v1_evk"])
    cv_v1_spon = float(S["cv_v1_spon"])

    cmap = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(alpha_list)))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    beta_ex = 0.3
    ib = int(np.argmin(np.abs(beta_grid - beta_ex)))
    ax = axes[0]
    for a, c in zip(alpha_list, cmap):
        logp = -beta_grid[ib] * (A_cost * r**a)
        logp -= np.nanmax(logp)
        p = np.exp(logp)
        p /= np.trapz(p, r)
        ax.plot(r, p, color=c, lw=1.6, label=fr"$\alpha={a:.1f}$")
    ax.set_xlabel(r"Firing rate $r$ (Hz)")
    ax.set_ylabel(r"$p^\star(r)$")
    ax.set_title(r"Gibbs distributions at $\beta\approx0.3$", pad=7)
    ax.text(-0.13, 1.03, "A", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.set_xlim(0, float(S["r_max"]))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for ia, (a, c) in enumerate(zip(alpha_list, cmap)):
        order = np.argsort(CV_arr[ia])
        ax.plot(CV_arr[ia, order], ICE_arr[ia, order], color=c, lw=1.6,
                label=fr"$\alpha={a:.1f}$")
    ax.axvline(1 / np.sqrt(3), color="k", ls="--", lw=1)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("CV")
    ax.set_ylabel(r"ICE = $H(r)/\langle c(r)\rangle$")
    ax.set_title("Bounded-support sublinear sweep", pad=7)
    ax.text(-0.13, 1.03, "B", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[2]
    ia07 = int(np.argmin(np.abs(alpha_list - 0.7)))
    order = np.argsort(CV_arr[ia07])
    cv = CV_arr[ia07, order]
    ice = ICE_arr[ia07, order]
    ax.plot(cv, ice, color="black", lw=2.0, label=r"$\alpha=0.7$")
    ice_evk = np.interp(cv_v1_evk, cv, ice)
    ice_spon = np.interp(cv_v1_spon, cv, ice)
    ax.scatter([cv_v1_evk], [ice_evk], s=95, color=COL_EVK, edgecolor="black",
               label="V1 evoked median", zorder=3)
    ax.scatter([cv_v1_spon], [ice_spon], s=95, color=COL_SPON, edgecolor="black",
               label="V1 spontaneous median", zorder=3)
    ax.axvline(1 / np.sqrt(3), color="k", ls="--", lw=1)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("CV")
    ax.set_ylabel("ICE")
    ax.set_title(r"V1 medians on $\alpha=0.7$", pad=7)
    ax.text(-0.13, 1.03, "C", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_both(fig, "FigureS13")
    plt.close(fig)


def make_s14() -> None:
    S = loadmat(FIG2 / "variance_decomp_CVP.mat", squeeze_me=True, struct_as_record=False)
    F_spn = np.asarray(S["F_spn"], dtype=float)
    F_evk = np.asarray(S["F_evk"], dtype=float)
    F_spn_mean = np.asarray(S["F_spn_mean"], dtype=float)
    F_evk_mean = np.asarray(S["F_evk_mean"], dtype=float)
    p_neuron = float(S["p_neuron"])
    p_trial = float(S["p_trial"])

    labels = ["across-neuron", "across-trial", "interaction"]
    colors = ["#8b8bd3", "#d98585", "#b4b4b4"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    bottom = np.zeros(2)
    yvals = np.vstack([F_spn_mean, F_evk_mean])
    for i, (lab, color) in enumerate(zip(labels, colors)):
        ax.bar([0, 1], yvals[:, i], bottom=bottom, color=color, edgecolor="white",
               label=lab)
        bottom += yvals[:, i]
    ax.set_xticks([0, 1], ["spontaneous", "evoked"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of total variance")
    ax.set_title("A: variance decomposition")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for row in F_spn:
        pass
    for isess in range(F_spn.shape[0]):
        ax.plot([0, 1], [F_spn[isess, 0], F_evk[isess, 0]], "-o",
                color=colors[0], lw=1.3, ms=5)
        ax.plot([2, 3], [F_spn[isess, 1], F_evk[isess, 1]], "-o",
                color=colors[1], lw=1.3, ms=5)
    ax.set_xticks([0, 1, 2, 3], ["spn", "evk", "spn", "evk"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("variance fraction")
    ax.set_title("B: paired sessions")
    ax.text(0.5, 1.02, fr"$F_N$ p={p_neuron:.3g}", ha="center")
    ax.text(2.5, 1.02, fr"$F_T$ p={p_trial:.3g}", ha="center")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Variance decomposition of V1 neuron x trial firing-rate matrices", y=1.03)
    fig.tight_layout()
    save_both(fig, "FigureS14")
    plt.close(fig)


def main() -> None:
    make_s12()
    make_s13()
    make_s14()


if __name__ == "__main__":
    main()
