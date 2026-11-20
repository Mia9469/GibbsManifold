"""multispecies_CVL_summary.py

Multi-species / multi-region CV_L and S_L summary figure.

Panels
------
A  S_L vs CV_L scatter — all datasets (evoked filled, spontaneous open)
   Identity curve S = CV²/(1+CV²) in black.
B  Paired spontaneous > evoked bar chart — curated 12 representative
   regions (Wilcoxon p-value stars, median ± IQR).
C  Proportion of brain regions where spon > evoked CV_L (Steinmetz atlas).

Data files required (pre-computed by companion scripts):
  awake_V1_CVL.npz            mouse VISp, Allen NPx awake
  Fig2_panels_CD_CVL_data.npz macaque V1 main-text dataset
  pvc8_CVL_results.npz        cat V1, evoked only
  steinmetz_CVL_results.npz   mouse 50 brain regions (Steinmetz 2019)

Note on pvc-11 (macaque, 3 monkeys):
  Spontaneous and evoked recordings are independent sessions without
  matched unit IDs, so per-neuron pairing is not possible.  We therefore
  show the *population distributions* in panel A only and exclude pvc-11
  from the paired panel B.

Usage:
    python multispecies_CVL_summary.py
"""
from __future__ import annotations
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon

BASE = Path(__file__).parent

# ── Helpers ───────────────────────────────────────────────────────────────────

def cv2_identity(cv):
    return cv ** 2 / (1.0 + cv ** 2)


def load_npz(path):
    p = Path(path)
    return np.load(str(p), allow_pickle=True) if p.exists() else None


def clean(arr):
    """Return finite values only."""
    a = np.asarray(arr, dtype=float)
    return a[np.isfinite(a)]


def bias_corrected_sparseness(r):
    N = len(r)
    if N < 2: return np.nan
    mu = r.mean(); mu2 = (r**2).mean()
    return np.nan if mu2 == 0 else (1 - mu**2/mu2) / (1 - 1/N)

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

SPECIES_COLOR = {"macaque": "#E07B39", "cat": "#4BAF72", "mouse": "#4C88C8"}

# ── Dataset catalogue ─────────────────────────────────────────────────────────
# Each entry: dict with label, species, region,
#   cvL_spon / cvL_evk (arrays or None),
#   sL_spon / sL_evk,
#   paired (bool) — whether spon & evk are per-neuron matched

datasets_A = []   # all datasets → panel A scatter
datasets_B = []   # paired datasets only → panel B

# ── 1. Macaque V1 main text (properly paired within-neuron) ───────────────────
d = load_npz(BASE / "Fig2_panels_CD_CVL_data.npz")
if d is not None:
    keys = list(d.keys())
    cs_k = next((k for k in keys if "spon" in k.lower() and "cv" in k.lower()), None)
    ce_k = next((k for k in keys if "evk"  in k.lower() and "cv" in k.lower()), None)
    ss_k = next((k for k in keys if "spon" in k.lower() and "sl" in k.lower()), None)
    se_k = next((k for k in keys if "evk"  in k.lower() and "sl" in k.lower()), None)
    if cs_k and ce_k:
        cs, ce = clean(d[cs_k]), clean(d[ce_k])
        ss = clean(d[ss_k]) if ss_k else cv2_identity(cs)
        se = clean(d[se_k]) if se_k else cv2_identity(ce)
        entry = dict(label="Macaque V1\n(main text)", species="macaque", region="V1",
                     cvL_spon=cs, cvL_evk=ce, sL_spon=ss, sL_evk=se, paired=True)
        datasets_A.append(entry)
        datasets_B.append(entry)
        print(f"[MACAQUE_V1_MAIN] N={len(cs)} cv_spon={np.median(cs):.2f} cv_evk={np.median(ce):.2f} spon>evk={'YES' if np.median(cs)>np.median(ce) else 'NO'}")

# ── 2. Cat V1 pvc-8 (evoked only) ────────────────────────────────────────────
d = load_npz(BASE / "pvc8_CVL_results.npz")
if d is not None:
    ce = clean(d["cvL_evk"])
    se = clean(d["sL_evk"])
    # FR filter: keep neurons with FR_evk > 0.3 Hz (removes silent cells)
    fr = d["cvL_evk"].copy()  # use FR field if present
    if "FR_evk" in d.files:
        fr_arr = np.asarray(d["FR_evk"], dtype=float)
        mask = np.isfinite(fr_arr) & (fr_arr > 0.3) & np.isfinite(d["cvL_evk"])
        ce = clean(d["cvL_evk"][mask])
        se = clean(d["sL_evk"][mask])
    entry = dict(label="Cat V1\n(pvc-8)", species="cat", region="V1",
                 cvL_spon=None, cvL_evk=ce, sL_spon=None, sL_evk=se, paired=False)
    datasets_A.append(entry)
    # cat goes to panel B as evoked-only bar (half bar, no spon)
    print(f"[CAT_V1_PVC8] N={len(ce)} cv_evk={np.median(ce):.2f} (evoked only, FR>0.3Hz)")

# ── 3. Macaque V1 pvc-11 (population-level only, no pairing) ─────────────────
pvc11 = load_npz(BASE / "pvc11_CVL_results.npz")
if pvc11 is not None:
    cs_all = clean(pvc11["cvL_spon"])
    ce_all = clean(pvc11["cvL_evk"])
    ss_all = clean(pvc11["sL_spon"])
    se_all = clean(pvc11["sL_evk"])
    # Add as population distributions to Panel A (no pairing)
    entry = dict(label="Macaque V1\n(pvc-11)", species="macaque", region="V1",
                 cvL_spon=cs_all, cvL_evk=ce_all,
                 sL_spon=ss_all, sL_evk=se_all, paired=False)
    datasets_A.append(entry)
    print(f"[PVC11_MACAQUE_V1] N={len(cs_all)} cv_spon={np.median(cs_all):.2f} cv_evk={np.median(ce_all):.2f} "
          f"NOTE: unpaired sessions — panel A only")

# ── 4. Mouse VISp Allen NPx awake (paired) ───────────────────────────────────
d = load_npz(BASE / "awake_V1_CVL.npz")
if d is not None:
    cs, ce = clean(d["cvL_spon"]), clean(d["cvL_evk"])
    entry = dict(label="Mouse VISp\n(Allen awake)", species="mouse", region="VISp",
                 cvL_spon=cs, cvL_evk=ce,
                 sL_spon=cv2_identity(cs), sL_evk=cv2_identity(ce), paired=True)
    datasets_A.append(entry)
    datasets_B.append(entry)
    print(f"[MOUSE_VISp_ALLEN] N={len(cs)} cv_spon={np.median(cs):.2f} cv_evk={np.median(ce):.2f} spon>evk={'YES' if np.median(cs)>np.median(ce) else 'NO'}")

# ── 5. Steinmetz 2019 — mouse, 50 brain regions ───────────────────────────────
# Panel A: all 50 areas (scatter only)
# Panel B: curated 12 representative regions (visual + non-visual)
PANEL_B_AREAS = ["VISp", "VISl", "VISpm", "VISam",   # mouse visual cortex
                 "CA1",  "DG",   "CA3",               # hippocampus
                 "MOs",  "ACA",  "ORBm",              # frontal/prefrontal
                 "AUD",  "CP"]                        # auditory, striatum

d = load_npz(BASE / "steinmetz_CVL_results.npz")
if d is not None:
    area_names = list(d["area_names"])
    area_code  = np.asarray(d["area_code"])
    for ai, area in enumerate(area_names):
        ce_arr = np.asarray(d[f"cvL_evk_{area}"], dtype=float)
        cs_arr = np.asarray(d[f"cvL_spon_{area}"], dtype=float)
        mask   = np.isfinite(ce_arr) & np.isfinite(cs_arr)
        if mask.sum() < 10:
            continue
        cs, ce = cs_arr[mask], ce_arr[mask]
        entry = dict(
            label=f"Mouse {area}\n(Steinmetz)",
            species="mouse", region=area,
            cvL_spon=cs, cvL_evk=ce,
            sL_spon=cv2_identity(cs), sL_evk=cv2_identity(ce),
            paired=True,  # within-session, within-neuron
        )
        datasets_A.append(entry)
        direction = "YES" if np.median(cs) > np.median(ce) else "NO"
        print(f"[STEINMETZ_{area}] N={mask.sum()} cv_spon={np.median(cs):.2f} cv_evk={np.median(ce):.2f} spon>evk={direction}")
        if area in PANEL_B_AREAS:
            datasets_B.append(entry)

print(f"\nPanel A: {len(datasets_A)} datasets | Panel B: {len(datasets_B)} paired datasets\n")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 7))
gs  = fig.add_gridspec(1, 3, width_ratios=[1.4, 2.0, 0.9],
                        wspace=0.35, left=0.06, right=0.97,
                        top=0.88, bottom=0.20)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC = fig.add_subplot(gs[2])

# ── Panel A: S_L vs CV_L scatter ──────────────────────────────────────────────
cv_range = np.linspace(0, 3.5, 400)
axA.plot(cv_range, cv2_identity(cv_range), "k-", lw=1.8, zorder=6,
         label="S = CV²/(1+CV²)")

MARKERS_BY_REGION = {}   # region → marker character
_mlist = ["o","s","^","D","v","P","X","h","*","<",">","8","p","H"]
_mi = 0
for ds in datasets_A:
    r = ds["region"]
    if r not in MARKERS_BY_REGION:
        MARKERS_BY_REGION[r] = _mlist[_mi % len(_mlist)]
        _mi += 1

rng = np.random.default_rng(42)
for ds in datasets_A:
    col = SPECIES_COLOR.get(ds["species"], "#888888")
    mk  = MARKERS_BY_REGION[ds["region"]]

    def _scatter(cv, sl, filled):
        if cv is None or len(cv) == 0: return
        n = len(cv)
        # macaque/cat get larger, more opaque markers so they stand out
        is_prominent = ds["species"] in ("macaque", "cat")
        sz     = 20 if is_prominent else 6
        alpha  = 0.70 if is_prominent else 0.35
        zorder = 5 if is_prominent else 3
        if n > 400 and not is_prominent:
            idx = rng.choice(n, 400, replace=False)
            cv, sl = cv[idx], sl[idx]
        fc = col if filled else "none"
        ec = col
        axA.scatter(cv, sl, s=sz, alpha=alpha, marker=mk,
                    facecolors=fc, edgecolors=ec, linewidths=0.6, zorder=zorder)

    _scatter(ds["cvL_evk"], ds["sL_evk"], filled=True)
    _scatter(ds["cvL_spon"], ds["sL_spon"], filled=False)

# species legend (compact)
legend_h = [plt.Line2D([0],[0], color="k", lw=1.8, label="S = CV²/(1+CV²)")]
for sp, col in SPECIES_COLOR.items():
    legend_h.append(mpatches.Patch(color=col, label=sp.capitalize()))
legend_h += [
    plt.Line2D([0],[0], marker="o", color="#888", mfc="#888", ms=6,
               linestyle="None", label="Evoked (filled)"),
    plt.Line2D([0],[0], marker="o", color="#888", mfc="none", ms=6,
               linestyle="None", label="Spontaneous (open)"),
]
axA.legend(handles=legend_h, fontsize=7.5, loc="lower right", frameon=False)
axA.set_xlabel("CV$_L$", fontsize=10)
axA.set_ylabel("$S_L$ (sparseness)", fontsize=10)
axA.set_xlim(-0.1, 3.7); axA.set_ylim(-0.03, 1.08)
axA.set_title("A  $S_L$ vs CV$_L$ — all species & regions", fontsize=10, pad=6)

# ── Panel B: Paired spon vs evoked, curated regions ───────────────────────────
n_B = len(datasets_B)
x   = np.arange(n_B)

for i, ds in enumerate(datasets_B):
    col = SPECIES_COLOR.get(ds["species"], "#888")
    mk  = MARKERS_BY_REGION.get(ds["region"], "o")
    cs  = clean(ds["cvL_spon"]) if ds["cvL_spon"] is not None else np.array([])
    ce  = clean(ds["cvL_evk"])

    def _bar(ax, xi, vals, filled):
        if len(vals) == 0: return
        med = np.median(vals); q1 = np.percentile(vals,25); q3 = np.percentile(vals,75)
        fc  = col if filled else "white"
        ec  = col
        ax.errorbar(xi, med, yerr=[[med-q1],[q3-med]],
                    fmt=mk, color=col, capsize=3, markersize=7,
                    markerfacecolor=fc, markeredgecolor=ec, markeredgewidth=1.2,
                    linewidth=1.4, zorder=4)
        return med

    xs = x[i] - 0.18
    xe = x[i] + 0.18
    ms = _bar(axB, xs, cs, filled=True)
    me = _bar(axB, xe, ce, filled=False)

    # arrow evoked → spontaneous
    if len(cs) > 0 and len(ce) > 0:
        axB.annotate("", xy=(xs, ms), xytext=(xe, me),
                     arrowprops=dict(arrowstyle="-|>", color=col,
                                     lw=0.9, mutation_scale=10))
        # significance
        n_p = min(len(cs), len(ce))
        if n_p >= 10:
            try:
                _, p = wilcoxon(cs[:n_p], ce[:n_p])
                sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
                if sig:
                    top = max(ms, me) + axB.get_ylim()[1]*0.03 if axB.get_ylim()[1]>0 else 0.15
                    axB.text(x[i], max(ms, me) + 0.12, sig,
                             ha="center", va="bottom", fontsize=9, color=col)
            except Exception:
                pass

axB.set_xticks(x)
labels_B = [ds["label"] for ds in datasets_B]
axB.set_xticklabels(labels_B, rotation=35, ha="right", fontsize=7.5)
axB.set_ylabel("CV$_L$  (median ± IQR)", fontsize=10)
axB.set_title("B  Spontaneous (●) vs Evoked (○) CV$_L$\n"
              "arrows: evoked → spontaneous", fontsize=10, pad=6)

# vertical dividers between species groups
species_order = [ds["species"] for ds in datasets_B]
for j in range(1, n_B):
    if species_order[j] != species_order[j-1]:
        axB.axvline(j - 0.5, color="#cccccc", lw=0.8, zorder=0)

sp_leg = []
seen = set()
for ds in datasets_B:
    sp = ds["species"]
    if sp not in seen:
        seen.add(sp)
        sp_leg.append(mpatches.Patch(color=SPECIES_COLOR[sp], label=sp.capitalize()))
axB.set_ylim(0, 4.5)   # clip extreme outliers for readability
axB.legend(handles=sp_leg, fontsize=8, loc="upper right", frameon=False)

# ── Panel C: Fraction of Steinmetz regions with spon > evoked ─────────────────
# Group by broad anatomical category
GROUPS = {
    "Visual\ncortex":   ["VISp","VISl","VISpm","VISam","VISrl","VISa"],
    "Hippocampus":      ["CA1","CA3","DG","SUB","POST"],
    "Frontal/\nPFC":    ["MOs","ACA","PL","ILA","ORB","ORBm"],
    "Thalamus":         ["TH","LGd","LP","MD","PO","POL","LD","VPL","MG"],
    "Striatum/\nBG":    ["CP","GPe","SCm","SCs","SCsg"],
    "Other":            [],   # catch-all
}

# collect Steinmetz results
stein_areas = {}
d = load_npz(BASE / "steinmetz_CVL_results.npz")
if d is not None:
    for area in list(d["area_names"]):
        ce_a = np.asarray(d[f"cvL_evk_{area}"], dtype=float)
        cs_a = np.asarray(d[f"cvL_spon_{area}"], dtype=float)
        mask = np.isfinite(ce_a) & np.isfinite(cs_a)
        if mask.sum() < 10: continue
        stein_areas[area] = np.median(cs_a[mask]) > np.median(ce_a[mask])

group_fracs = {}
group_ns    = {}
for gname, area_list in GROUPS.items():
    if gname == "Other":
        listed = [a for gl in GROUPS.values() for a in gl]
        subset = {a: v for a,v in stein_areas.items() if a not in listed}
    else:
        subset = {a: stein_areas[a] for a in area_list if a in stein_areas}
    if subset:
        group_fracs[gname] = np.mean(list(subset.values()))
        group_ns[gname]    = len(subset)

if group_fracs:
    gnames = list(group_fracs.keys())
    fracs  = [group_fracs[g] for g in gnames]
    ns     = [group_ns[g]    for g in gnames]
    colors_C = ["#4C88C8"]*len(gnames)

    bars = axC.barh(np.arange(len(gnames)), fracs, color=colors_C,
                    edgecolor="white", height=0.6, zorder=3)
    axC.axvline(0.5, color="k", lw=0.8, ls="--", zorder=4)
    for j, (f, n) in enumerate(zip(fracs, ns)):
        axC.text(min(f+0.03, 0.97), j, f"{f:.0%} (n={n})",
                 va="center", fontsize=7.5)
    axC.set_yticks(np.arange(len(gnames)))
    axC.set_yticklabels(gnames, fontsize=8)
    axC.set_xlim(0, 1.15)
    axC.set_xlabel("Fraction of regions\nspon > evoked CV$_L$", fontsize=9)
    axC.set_title("C  Consistency across\nbrain regions (Steinmetz)", fontsize=10, pad=6)
    axC.set_xticks([0, 0.5, 1.0])
    axC.set_xticklabels(["0%", "50%", "100%"], fontsize=8)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Multi-species / multi-region: spontaneous activity consistently "
    "shows higher CV$_L$ than evoked\n"
    "macaque V1 (CRCNS pvc-11 + main text) · cat V1 (pvc-8) · "
    "mouse 50 regions (Steinmetz 2019) · mouse VISp (Allen NPx awake)",
    fontsize=9, y=0.97
)

out_pdf = BASE / "FigureS_multispecies.pdf"
out_png = BASE / "FigureS_multispecies.png"
fig.savefig(str(out_pdf), bbox_inches="tight")
fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

dst_dir = BASE.parent / "updated" / "figures"
dst_dir.mkdir(parents=True, exist_ok=True)
for src in [out_pdf, out_png]:
    shutil.copy2(str(src), str(dst_dir / src.name))
    print(f"Copied to: {dst_dir / src.name}")

print("\n=== done ===")
