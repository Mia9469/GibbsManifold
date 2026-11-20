"""
Fig 1C: four panels illustrating the Gibbs manifold geometry.

  (i)   H–C frontier  (H in bits, parametric linear-cost Gibbs manifold)
  (ii)  S–CV algebraic identity   (distribution-independent, no reference lines)
  (iii) ICE–CV along the Gibbs-optimal manifold  (bits / unit cost)
  (iv)  Exact count agreement P_agree vs CV  (Poisson, Supp §2.3)

Output: Figure1C_panels.pdf / .png  — PDF uses TrueType (Type 42) fonts
so text remains editable in Illustrator / Affinity Designer / etc.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import i0  # modified Bessel I_0

# ---- Editable-PDF settings: keep text as text (TrueType), not as outlines ----
mpl.rcParams['pdf.fonttype']       = 42
mpl.rcParams['ps.fonttype']        = 42
mpl.rcParams['svg.fonttype']       = 'none'
mpl.rcParams['text.usetex']        = False          # keeps text strings editable
mpl.rcParams['font.family']        = 'sans-serif'
mpl.rcParams['font.sans-serif']    = ['Helvetica', 'Arial', 'DejaVu Sans']

LOG2 = np.log(2.0)   # conversion factor nats → bits

# =====================================================================
# Truncated exponential on [0, rmax]  with parameter β;  k = β·rmax
# =====================================================================

def mu(k):
    """E[r]/rmax for truncated exponential on [0,1]."""
    if k > 1e-8:
        return (1.0 - k / np.expm1(k)) / k
    return 0.5 - k / 12                         # Taylor for k ≈ 0

def var_norm(k):
    """Var(r)/rmax² for truncated exponential on [0,1]."""
    if k < 1e-6:
        return 1/12 - k**2 / 180
    Er2 = (2 - (k**2 + 2*k) / np.expm1(k)) / k**2
    return Er2 - mu(k)**2

def CV_of_k(k):
    return np.sqrt(var_norm(k)) / mu(k)

def H_nats(k, rmax=1.0):
    """Differential entropy in nats."""
    return np.log(rmax * (1 - np.exp(-k)) / k) + (1 - k / np.expm1(k))

def H_bits(k, rmax=1.0):
    """Differential entropy in bits."""
    return H_nats(k, rmax) / LOG2

def ICE_bits(k, rmax=1.0):
    """ICE = H / E[r]  in bits / unit cost."""
    return H_bits(k, rmax) / (rmax * mu(k))

# =====================================================================
# Parameters
# =====================================================================

R_MAX      = 10.0                              # rmax ≳ e keeps H > 0
k_ice      = np.logspace(-2, 1.0, 400)         # k ≈ 0.01 … 10  (panel iii; reaches ICE peak at k≈r_max)
k_frontier = np.logspace(-3, 1.2, 600)         # wider range      (panel i)

# =====================================================================
# Compute
# =====================================================================

cv_ice      = np.array([CV_of_k(k)           for k in k_ice])
ice_vals    = np.array([ICE_bits(k, R_MAX)   for k in k_ice])

H_frontier  = np.array([H_bits(k, R_MAX)     for k in k_frontier])
C_frontier  = R_MAX * np.array([mu(k)        for k in k_frontier])
CV_frontier = np.array([CV_of_k(k)           for k in k_frontier])

# =====================================================================
# Plot
# =====================================================================

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

# ---------- (i) H–C frontier  (bits) ----------
ax = axes[0]
ax.fill_between(C_frontier, 0, H_frontier, alpha=0.12, color='gray')
ax.plot(C_frontier, H_frontier, 'k-', lw=2.2, zorder=2,
        label=r'$H^\star(C)$ (Gibbs manifold)')
sc = ax.scatter(C_frontier[::25], H_frontier[::25],
                c=CV_frontier[::25], cmap='viridis',
                s=22, zorder=3, edgecolor='none')
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('CV', fontsize=9)

# Arrow: FROM low CV (large C, high H) TOWARD high CV (small C, lower H)
# Convention used in ax.annotate: the arrow HEAD is at `xy`, tail at `xytext`.
i_tail, i_head = 120, 420   # i_tail: low-CV end; i_head: high-CV end
ax.annotate('',
            xy=(C_frontier[i_head], H_frontier[i_head]),       # arrow head
            xytext=(C_frontier[i_tail], H_frontier[i_tail]),   # arrow tail
            arrowprops=dict(arrowstyle='->', color='crimson', lw=1.8))
ax.text(0.05, 0.88, 'increasing\nmetabolic\npressure',
        transform=ax.transAxes, color='crimson', fontsize=9, va='top')
ax.text(0.55, 0.18, 'suboptimal',
        transform=ax.transAxes, color='gray', fontsize=10, style='italic')
ax.set_xlabel(r'Metabolic budget $C = \mathbb{E}[r]$')
ax.set_ylabel(r'Entropy $H$ (bits)')
ax.set_title(r'(i)  $H$–$C$ frontier')

# ---------- (ii) S–CV algebraic identity  (no reference lines) ----------
# This relation is an algebraic identity S = CV²/(1+CV²) independent of the
# underlying response distribution, so uniform / exponential markers are
# meaningless here and are deliberately omitted.
ax = axes[1]
cv_range = np.linspace(0, 3, 400)
S_curve  = cv_range**2 / (1 + cv_range**2)
ax.plot(cv_range, S_curve, 'k-', lw=2.2,
        label=r'$S = \mathrm{CV}^2 / (1+\mathrm{CV}^2)$')
ax.set_xlim(0, 3); ax.set_ylim(0, 1)
ax.set_xlabel(r'CV')
ax.set_ylabel(r'Sparseness $S$')
ax.set_title(r'(ii)  $S$–CV (algebraic identity)')
ax.legend(loc='lower right', fontsize=9)

# ---------- (iii) ICE–CV  (bits, same curve reused as Gibbs overlay in Fig 4G) ----------
ax = axes[2]
ax.plot(cv_ice, ice_vals, 'k-', lw=2.2, label='Gibbs-optimal manifold')
ax.axvline(1/np.sqrt(3), color='purple', ls='--', lw=0.9, alpha=0.6,
           label=r'Uniform limit ($\mathrm{CV}=1/\sqrt{3}$)')
ax.axvline(1.0,          color='teal',   ls='--', lw=0.9, alpha=0.6,
           label=r'Exponential limit ($\mathrm{CV}=1$)')
ax.set_xlabel(r'CV')
ax.set_ylabel(r'ICE $= H / \mathbb{E}[r]$ (bits / unit cost)')
ax.set_title(r'(iii)  ICE–CV  ($r_{\max}=%g$)' % R_MAX)
ax.set_ylim(0.6, 1.5)
ax.legend(loc='lower right', fontsize=8)

# ---------- (iv) P_agree–CV  (Poisson) ----------
# Supp §2.3: trial-to-trial response agreement P_agree = P(X=Y) for X,Y iid
# Pois(λ).  With CV = 1/√λ  ⇒  λ = 1/CV².  P_agree = e^{-2λ} I_0(2λ), which
# is monotonically increasing in CV.  This is exact count agreement, not decoder reliability.
ax = axes[3]
cv_rel = np.linspace(0.3, 3, 400)
lam    = 1.0 / cv_rel**2
P_agr  = np.exp(-2*lam) * i0(2*lam)
ax.plot(cv_rel, P_agr, 'k-', lw=2.2,
        label=r'Poisson: $P_{\mathrm{agree}} = e^{-2\lambda} I_0(2\lambda),\ \lambda = 1/\mathrm{CV}^2$')
ax.set_xlabel(r'CV')
ax.set_ylabel(r'$P_{\mathrm{agree}}$')
ax.set_title(r'(iv)  Exact count agreement vs CV (Poisson-only)')
ax.legend(loc='lower right', fontsize=8)

for a in axes:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.tight_layout()

# Editable-PDF export: vector graphics, TrueType text, no rasterization.
plt.savefig('Figure1C_panels.pdf', bbox_inches='tight',
            metadata={'Creator': 'matplotlib — Gibbs manifold Fig 1C'})
plt.savefig('Figure1C_panels.png', dpi=200, bbox_inches='tight')

# =====================================================================
# Fig 4G: Gibbs-optimal ICE–CV manifold as a reference curve for empirical
# overlay.  Same analytical curve as panel (iii), but drawn standalone
# with a wider y-range (0–3 bits/unit cost) to accommodate experimental
# data points whose absolute ICE may exceed the linear-cost ceiling
# (~1.44 bits at k = r_max) due to sub-linear cost or different info/cost
# operationalisations (Methods, ICE table).
# =====================================================================

fig4g, ax = plt.subplots(figsize=(5.2, 4.2))
ax.plot(cv_ice, ice_vals, 'k-', lw=2.4,
        label='Gibbs-optimal manifold (linear cost)')
ax.axvline(1/np.sqrt(3), color='purple', ls='--', lw=0.9, alpha=0.6,
           label=r'Uniform limit ($\mathrm{CV}=1/\sqrt{3}$)')
ax.axvline(1.0,          color='teal',   ls='--', lw=0.9, alpha=0.6,
           label=r'Exponential limit ($\mathrm{CV}=1$)')
ax.set_xlabel(r'CV')
ax.set_ylabel(r'ICE $= H / \mathbb{E}[r]$ (bits / unit cost)')
ax.set_title(r'Fig 4G  Gibbs-optimal ICE–CV reference  ($r_{\max}=%g$)' % R_MAX)
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 3.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', fontsize=9)
fig4g.tight_layout()

fig4g.savefig('Figure4G_reference.pdf', bbox_inches='tight',
              metadata={'Creator': 'matplotlib — Gibbs manifold Fig 4G reference'})
fig4g.savefig('Figure4G_reference.png', dpi=200, bbox_inches='tight')

plt.show()
