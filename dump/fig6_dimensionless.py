"""
Fig 6 panel A — dimensionless ICE-CV overlay across single-neuron model,
recurrent network, and macaque V1.

The universal Gibbs frontier is plotted as:
  - solid black line: linear cost on [0, r_max], CV in [1/sqrt(3), 1]
  - shaded fan: sublinear cost c(r) ~ r^alpha for alpha in [0.4, 1],
    extending the frontier to higher CV (Supplementary Section 1.6)

No per-system rate-conversion factor enters this panel: ICE = H/C is
intrinsic when both H and C are computed at the same analysis-window
duration (cancellation), and CV is dimensionless.

Per-system data are loaded from CSV files (or filled in by hand below).
Each row = one measurement = (CV, ICE).  Replace placeholders with the
actual exported summary statistics from the per-system pipelines.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import brentq

mpl.rcParams['pdf.fonttype']    = 42
mpl.rcParams['ps.fonttype']     = 42
mpl.rcParams['svg.fonttype']    = 'none'
mpl.rcParams['text.usetex']     = False
mpl.rcParams['font.family']     = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

LOG2 = np.log(2.0)

# =====================================================================
# Theoretical Gibbs frontier
# =====================================================================
#
# Linear cost on [0, r_max]: truncated exponential family (see Fig1c.py)
#   CV in [1/sqrt(3), 1], ICE peaks at ~1.44 bits/J for r_max = 10
#
# Sublinear cost c(r) = b * r^alpha on [0, infinity):
#   Gibbs:  p*(r) = alpha * (beta b)^(1/alpha) / Gamma(1/alpha) * exp(-beta b r^alpha)
#   CV(alpha) = sqrt( Gamma(1/a) Gamma(3/a) / Gamma(2/a)^2 - 1 )      (Eq. cv-alpha)
#   E[c]      = 1 / (alpha beta)
#   H[p*]     = log[Gamma(1/a)/a] - (1/a) log(beta b) + 1/a            (in nats)
#   ICE       = H / E[c]   in nats per cost unit; convert to bits.
#
# At fixed r_max=10 (linear regime cap) and matching baseline cost
# c0 = 2.4 used in result5.tex, the absolute scale of ICE is set by
# the per-system analysis-window duration via the H rate conversion
# (a constant prefactor that cancels in the dimensionless plot).

R_MAX = 10.0  # firing-rate ceiling (Hz) for the linear-cost regime

def cv_of_alpha(a):
    g1, g2, g3 = gammaln(1/a), gammaln(2/a), gammaln(3/a)
    return float(np.sqrt(np.exp(g1 + g3 - 2*g2) - 1))

def alpha_of_cv(cv_t, lo=0.05, hi=5.0):
    return brentq(lambda a: cv_of_alpha(a) - cv_t, lo, hi)

def ice_at_cv_sublinear(cv_t, C=1.0, b=1.0):
    """ICE (nats / cost-unit) on the sublinear-cost frontier at a given CV."""
    a = alpha_of_cv(cv_t)
    beta = 1.0 / (a * C)
    H_nats = gammaln(1/a) - np.log(a) - (1/a)*np.log(beta*b) + 1/a
    return (H_nats / C) / LOG2  # convert to bits

# Linear-cost frontier (truncated exponential, same as Fig1c.py panel iii)
def mu(k):
    return (1.0 - k/np.expm1(k))/k if k > 1e-8 else 0.5 - k/12

def var_norm(k):
    if k < 1e-6: return 1/12 - k**2/180
    Er2 = (2 - (k**2 + 2*k)/np.expm1(k)) / k**2
    return Er2 - mu(k)**2

def cv_linear(k):
    return np.sqrt(var_norm(k)) / mu(k)

def H_linear_nats(k, rmax=R_MAX):
    return np.log(rmax * (1 - np.exp(-k))/k) + (1 - k/np.expm1(k))

def ice_linear(k, rmax=R_MAX):
    return H_linear_nats(k, rmax) / (rmax * mu(k)) / LOG2

# Build the linear-cost frontier curve in (CV, ICE) space
k_lin   = np.logspace(-2, 1.0, 400)
cv_lin  = np.array([cv_linear(k) for k in k_lin])
ice_lin = np.array([ice_linear(k) for k in k_lin])

# Build the sublinear-cost extension (CV in [1, ~3], parametric in alpha)
cv_sub  = np.linspace(1.0, 3.0, 200)
ice_sub = np.array([ice_at_cv_sublinear(c, C=1.0, b=1.0) for c in cv_sub])

# =====================================================================
# Per-system empirical data (FILL IN with actual exported values)
# =====================================================================
# Each entry = numpy array of (CV, ICE) pairs.
# Placeholder values are illustrative only; replace with values printed
# by your analysis pipelines (figure2_new.m for V1; equivalent scripts
# for single-neuron and recurrent network).

# Single-neuron model: learning trajectory, e.g. early -> late epochs
sn_CV  = np.array([0.55, 0.62, 0.71, 0.83, 0.95])
sn_ICE = np.array([0.82, 0.95, 1.08, 1.22, 1.35])

# Recurrent network: directed migration toward higher CV
rn_CV  = np.array([0.85, 1.05, 1.32, 1.58, 1.80])
rn_ICE = np.array([1.10, 1.25, 1.45, 1.62, 1.78])

# Macaque V1: spontaneous (open) and evoked (filled)
v1_spon_CV  = np.array([1.20, 1.35, 1.42, 1.48, 1.55])
v1_spon_ICE = np.array([1.50, 1.62, 1.70, 1.76, 1.82])
v1_evok_CV  = np.array([0.95, 1.05, 1.15, 1.22, 1.30])
v1_evok_ICE = np.array([1.20, 1.32, 1.42, 1.48, 1.55])

# =====================================================================
# Plot
# =====================================================================

fig, ax = plt.subplots(figsize=(6.0, 5.0))

# Universal Gibbs frontier: linear-cost segment (solid)
ax.plot(cv_lin, ice_lin, 'k-', lw=2.4,
        label='Gibbs frontier (linear cost)', zorder=3)

# Sublinear extension: dashed grey curve + shaded fan
ax.plot(cv_sub, ice_sub, color='0.35', ls='--', lw=1.6,
        label=r'Sublinear-cost extension ($\alpha\in[0.4,1]$)', zorder=2)
ax.fill_between(cv_sub, 0, ice_sub, color='0.85', alpha=0.5, zorder=1)

# Single-neuron: dotted trajectory with markers
ax.plot(sn_CV, sn_ICE, 'o:', color='tab:blue',  lw=1.4, ms=6,
        label='Single-neuron model', zorder=4)

# Recurrent network: solid trajectory with markers
ax.plot(rn_CV, rn_ICE, 's-', color='tab:green', lw=1.4, ms=6,
        label='Recurrent network', zorder=4)

# Macaque V1: spontaneous (open) + evoked (filled)
ax.scatter(v1_spon_CV, v1_spon_ICE, marker='o', s=70, facecolor='none',
           edgecolor='tab:red', lw=1.6, label='V1 spontaneous', zorder=5)
ax.scatter(v1_evok_CV, v1_evok_ICE, marker='o', s=55, color='tab:orange',
           edgecolor='black', lw=0.6, label='V1 evoked', zorder=5)

# Axis bounds & labels
ax.set_xlim(0.4, 2.5)
ax.set_ylim(0.5, 3.0)
ax.set_xlabel(r'CV $= \sigma/\mu$ (dimensionless)')
ax.set_ylabel(r'ICE $= H/C$ (bits / J, intrinsic)')
ax.set_title('Fig 6A  Dimensionless ICE\u2013CV overlay across three systems')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', fontsize=8, framealpha=0.92)

fig.tight_layout()
fig.savefig('Figure6A_dimensionless.pdf', bbox_inches='tight',
            metadata={'Creator': 'matplotlib \u2014 Gibbs manifold Fig 6A overlay'})
fig.savefig('Figure6A_dimensionless.png', dpi=200, bbox_inches='tight')
plt.show()
