"""
Supplementary Figure S6 -- Anchor sensitivity of the Fig. 6 R_0 calibration.

Fig. 6 anchors the theoretical Gibbs frontier at the minimum observed firing rate
(per system) to fix the single scale factor R_0 required to express bits/s in
absolute units.  Here we show that changing the anchor (min, median, or max rate)
simply rescales R_0 multiplicatively, leaving the shape of the data trajectory
invariant but shifting the frontier curve uniformly in bits/s.

The two panels mirror the two rows of main-text Fig. 6: I-C on top, ICE-C on
bottom.  Same data and same three anchor curves in both panels.

Output
------
  FigureS6_anchor_sensitivity.pdf  --  two-panel figure:
    (A) I-C panel: V1 data with the theoretical frontier under three anchors.
    (B) ICE-C panel: V1 data with the corresponding ICE frontier.
"""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.family']  = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9

# --- Load pooled V1 data -----------------------------------------------------
ROOT = Path('../')
MAT  = ROOT / 'Fig2' / 'unified_v1_Hrho_ICE.mat'

m = sio.loadmat(str(MAT), squeeze_me=True, struct_as_record=False)
FR = np.concatenate([np.asarray(s.FR).ravel() for s in m['results'].v1v2])
C  = np.concatenate([np.asarray(s.C ).ravel() for s in m['results'].v1v2])
I  = np.concatenate([np.asarray(s.mi).ravel() for s in m['results'].v1v2])

C0, C1 = 2.4, 0.37  # Eq (entropy_rate) offsets (matches main text)
keep = C > C0
C, I, FR = C[keep], I[keep], FR[keep]

# --- Frontier and anchor utilities ------------------------------------------
def frontier(C_vec, R0):
    return (R0 / np.log(2)) * (1.0 + np.log((C_vec - C0) / C1))

def R0_for_anchor(C_a, I_a):
    return I_a * np.log(2) / (1.0 + np.log((C_a - C0) / C1))

anchors = {
    'Minimum rate':   np.argmin(FR),
    'Median rate':    np.argsort(FR)[len(FR)//2],
    'Maximum rate':   np.argmax(FR),
}
colors = {'Minimum rate': 'C0', 'Median rate': 'C2', 'Maximum rate': 'C3'}
R0_vals = {lab: R0_for_anchor(C[i], I[i]) for lab, i in anchors.items()}

# --- Plot --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
ax1, ax2 = axes

C_grid = np.linspace(C.min()*0.99, C.max()*1.01, 400)

# Panel A: I-C with three frontiers (mirrors Fig 6 top row)
ax1.scatter(C, I, s=8, c='0.5', alpha=0.4, edgecolor='none', label='V1 data')
for lab, R0 in R0_vals.items():
    I_front = frontier(C_grid, R0)
    ax1.plot(C_grid, I_front, lw=1.8, color=colors[lab],
             label=f'{lab} anchor: $R_0$={R0:.2f}')
    idx = anchors[lab]
    ax1.plot(C[idx], I[idx], 'o', ms=9, mfc='none', mec=colors[lab], mew=2)
ax1.set_xlabel('Metabolic cost  $C$  (arb. units)')
ax1.set_ylabel(r'Mutual information  $I$  (bits/s)')
ax1.set_title(r'(A)  $I$--$C$ under three anchor choices')
ax1.legend(frameon=False, fontsize=8, loc='upper left')

# Panel B: ICE-C with three frontiers (mirrors Fig 6 bottom row)
ICE_data = I / C
ax2.scatter(C, ICE_data, s=8, c='0.5', alpha=0.4, edgecolor='none', label='V1 data')
for lab, R0 in R0_vals.items():
    ICE_front = frontier(C_grid, R0) / C_grid
    ax2.plot(C_grid, ICE_front, lw=1.8, color=colors[lab],
             label=f'{lab} anchor: $R_0$={R0:.2f}')
    idx = anchors[lab]
    ax2.plot(C[idx], ICE_data[idx], 'o', ms=9, mfc='none', mec=colors[lab], mew=2)
ax2.set_xlabel('Metabolic cost  $C$  (arb. units)')
ax2.set_ylabel(r'$\mathrm{ICE} = I/C$  (bits per unit cost)')
ax2.set_title(r'(B)  $\mathrm{ICE}$--$C$ under three anchor choices')
ax2.legend(frameon=False, fontsize=8, loc='upper right')

fig.tight_layout()

OUT = ROOT / 'updated' / 'figures' / 'FigureS6_anchor_sensitivity.pdf'
fig.savefig(str(OUT), bbox_inches='tight')
print(f'Wrote {OUT}')
print('R0 values:', {k: round(v, 3) for k, v in R0_vals.items()})

# Diagnostic gap stats (no longer plotted, but kept for the supp table)
for lab, R0 in R0_vals.items():
    gap = I - frontier(C, R0)
    print(f'  {lab}: median I_gap={np.median(gap):+.2f} bits/s, '
          f'{(gap >= 0).mean()*100:.1f}% above frontier')
