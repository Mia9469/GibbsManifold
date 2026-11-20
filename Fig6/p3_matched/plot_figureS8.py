"""
Re-render the Fig.~S8 RAE-vs-RE Pareto plot from the aggregated sweep
summary p3_summary.csv.  The x-axis is cropped at RE>=0.60 to focus
on the region where all three methods (Null, SparseNet, CVM) have
data; Null's under-regularised tail down to RE~0.37 is omitted from
the plot and discussed in the caption.

Output: ../../updated/figures/FigureS8.pdf
"""
from pathlib import Path
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.family']  = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9

ROOT = Path(__file__).parent
CSV  = ROOT / 'results' / 'eval' / 'p3_summary.csv'
OUT  = ROOT.parent.parent / 'updated' / 'figures' / 'FigureS8.pdf'

# Colour / label conventions (match main-text Fig 7)
COLOURS = {'null': '0.45', 'cauchy': '#1f77b4', 'cvm': '#2ca02c'}
LABELS  = {'null': 'Null  (L1 sweep $\\lambda$)',
           'cauchy': 'SparseNet  (Cauchy $\\alpha_s$)',
           'cvm': 'CV-max  (per-$\\rho$ sub-curves)'}

RE_MIN = 0.60  # crop x-axis here

def load_summary(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    rows = load_summary(CSV)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    # Group by method, then for CVM further by submethod / cv-ratio rho
    by_method = {}
    for r in rows:
        m = r['method']
        by_method.setdefault(m, []).append(r)

    # ---- Null (single sweep) ----
    if 'null' in by_method:
        grp = sorted(by_method['null'], key=lambda r: float(r['RE_mean']))
        x = np.array([float(r['RE_mean'])         for r in grp])
        y = np.array([float(r['ICE_log_a_mean']) for r in grp])
        yerr = np.array([float(r['ICE_log_a_std']) for r in grp])
        # plot every point with marker, but show line ONLY in [RE_MIN, ...]
        m_show = x >= RE_MIN
        ax.errorbar(x[m_show], y[m_show], yerr=yerr[m_show],
                    fmt='o-', ms=6, lw=1.6, capsize=2,
                    color=COLOURS['null'], label=LABELS['null'])

    # ---- SparseNet ----
    if 'cauchy' in by_method:
        grp = sorted(by_method['cauchy'], key=lambda r: float(r['RE_mean']))
        x = np.array([float(r['RE_mean'])         for r in grp])
        y = np.array([float(r['ICE_log_a_mean']) for r in grp])
        yerr = np.array([float(r['ICE_log_a_std']) for r in grp])
        m_show = x >= RE_MIN
        ax.errorbar(x[m_show], y[m_show], yerr=yerr[m_show],
                    fmt='s-', ms=6, lw=1.6, capsize=2,
                    color=COLOURS['cauchy'], label=LABELS['cauchy'])

    # ---- CVM v2 (absolute-RE gate) sub-curves indexed by cv_ratio rho ----
    # Each sub-curve fixes rho and traces RE-target tau in {0.40, 0.50, 0.55, 0.60}.
    if 'cvmax_v2' in by_method:
        cvm_grp = by_method['cvmax_v2']
        sub_groups = {}
        for r in cvm_grp:
            sub_groups.setdefault(r['submethod'], []).append(r)
        ordered_subs = sorted(sub_groups.keys(),
                              key=lambda s: float(s.split('cv')[-1]))
        ramp = plt.cm.Greens(np.linspace(0.45, 0.92, max(1, len(ordered_subs))))
        for ic, sub in enumerate(ordered_subs):
            grp = sorted(sub_groups[sub], key=lambda r: float(r['RE_mean']))
            x = np.array([float(r['RE_mean'])         for r in grp])
            y = np.array([float(r['ICE_log_a_mean']) for r in grp])
            yerr = np.array([float(r['ICE_log_a_std']) for r in grp])
            m_show = x >= RE_MIN
            if m_show.sum() == 0:
                continue
            rho_val = sub.split('cv')[-1]
            lab = f'CV-max  ($\\rho={rho_val}$)'
            ax.errorbar(x[m_show], y[m_show], yerr=yerr[m_show],
                        fmt='^-', ms=6, lw=1.4, capsize=2,
                        color=ramp[ic], label=lab)

    # Vertical dashed line at the matched-RE comparison point
    ax.axvline(0.60, ls='--', color='0.35', lw=0.9, zorder=0)
    ax.text(0.602, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 14.0,
            ' matched RE = 0.60', va='top', fontsize=8, color='0.35')

    ax.set_xlim(left=RE_MIN)
    ax.set_xlabel('Reconstruction error  RE')
    ax.set_ylabel(r'RAE score  $= -\log_2(\mathrm{RE})/\bar{a}$')
    ax.set_title('CV-max RAE--RE Pareto comparison  (sweep aggregated; $n=3$ seeds per point)')
    ax.legend(frameon=False, loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches='tight')
    print(f'Wrote {OUT}')

    # Also report the cropped points for verification
    if 'null' in by_method:
        cropped = [r for r in by_method['null'] if float(r['RE_mean']) < RE_MIN]
        print(f'Null points hidden by RE>={RE_MIN}: '
              + ', '.join(f"lambda={r['knob']} -> RE={float(r['RE_mean']):.3f}"
                          for r in cropped))

if __name__ == '__main__':
    main()
