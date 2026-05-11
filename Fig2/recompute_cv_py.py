"""
Independent re-computation of V1 spon vs evoked statistics directly from
105l001p16.mat, completely bypassing the MATLAB code path.  Reports both
- CV_P (per-trial population CV, std across neurons / mean across neurons)
- CV_L (per-neuron lifetime CV, std across trials / mean across trials)
plus mean firing rate and sparseness, and per-stimulus breakdowns for
the evoked condition.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.io as sio

MAT = Path("/Users/mia/Library/Mobile Documents/com~apple~CloudDocs/GitHub/GibbsManifold/Fig2/105l001p16.mat")
N_V1 = 159
DUR_SPON   = 1.5    # seconds (1500 bins x 1 ms)
DUR_DRIVEN = 1.28

print(f"[load] {MAT}")
m = sio.loadmat(str(MAT), squeeze_me=False, struct_as_record=False)
nd = m["neuralData"][0, 0]
rasters = nd.spikeRasters
stim    = nd.stim.ravel().astype(int)
n_trials_total = rasters.shape[0]
print(f"[load] {n_trials_total} trials")

# Build per-trial rate matrices: rates_spon [n_v1 x n_spon], rates_evk [n_v1 x n_evk]
spon_rates = []     # list of length-N_V1 vectors
evk_rates  = []
evk_stim   = []
for i in range(n_trials_total):
    r = rasters[i, 0]   # csc_matrix shape (159, T)
    if r.shape[1] == 1500:
        rates = np.asarray(r.sum(axis=1)).ravel() / DUR_SPON
        spon_rates.append(rates)
    elif r.shape[1] == 1280:
        rates = np.asarray(r.sum(axis=1)).ravel() / DUR_DRIVEN
        evk_rates.append(rates)
        evk_stim.append(stim[i])
spon_rates = np.array(spon_rates).T   # [n_v1, n_spon_trials]
evk_rates  = np.array(evk_rates).T    # [n_v1, n_evk_trials]
evk_stim   = np.asarray(evk_stim)
print(f"[shape] spon_rates {spon_rates.shape}, evk_rates {evk_rates.shape}")
print(f"[stim]  evoked stim breakdown: {sorted(set(evk_stim.tolist()))}, counts per stim: "
      + ", ".join(f"{s}:{int(np.sum(evk_stim==s))}" for s in sorted(set(evk_stim.tolist()))))


# --- per-trial population CV and S, mean ---
def per_trial_stats(R, label):
    # R: [n_v1, n_trials]
    mean_r  = R.mean(axis=0)        # [n_trials], mean across neurons in each trial
    std_r   = R.std(axis=0)
    cv_P    = std_r / mean_r
    # Vinje-Gallant: S = (1 - <r>^2/<r^2>) / (1 - 1/N)
    second  = (R**2).mean(axis=0)
    S_P     = (1 - mean_r**2 / second) / (1 - 1/N_V1)
    print(f"\n[{label}] per-trial population stats (n={R.shape[1]} trials)")
    print(f"  mean_FR (Hz): mean={mean_r.mean():.3f}  median={np.median(mean_r):.3f}")
    print(f"  CV_P        : mean={cv_P.mean():.4f}    median={np.median(cv_P):.4f}")
    print(f"  S_P         : mean={S_P.mean():.4f}    median={np.median(S_P):.4f}")
    return mean_r, cv_P, S_P

mean_spon, cv_P_spon, S_P_spon = per_trial_stats(spon_rates, "spontaneous")
mean_evk,  cv_P_evk,  S_P_evk  = per_trial_stats(evk_rates,  "evoked    ")


# --- per-neuron LIFETIME CV (across trials within condition) ---
def lifetime_stats(R, label):
    # R: [n_v1, n_trials]; CV_L per neuron = std across trials / mean across trials
    mu  = R.mean(axis=1)           # [n_v1]
    sd  = R.std(axis=1)
    cv_L = sd / np.where(mu > 1e-9, mu, np.nan)
    second = (R**2).mean(axis=1)
    S_L = (1 - mu**2 / second) / (1 - 1/R.shape[1])  # bias correction over trials
    print(f"\n[{label}] per-neuron lifetime stats (n={R.shape[0]} neurons)")
    print(f"  mean_FR (Hz): mean={mu.mean():.3f}  median={np.median(mu):.3f}")
    print(f"  CV_L        : mean={np.nanmean(cv_L):.4f}  median={np.nanmedian(cv_L):.4f}")
    print(f"  S_L         : mean={np.nanmean(S_L):.4f}  median={np.nanmedian(S_L):.4f}")
    return mu, cv_L, S_L

mu_n_spon, cv_L_spon, S_L_spon = lifetime_stats(spon_rates, "spontaneous")
mu_n_evk,  cv_L_evk,  S_L_evk  = lifetime_stats(evk_rates,  "evoked    ")


# --- per-stimulus breakdown for evoked (CV_P within each stim) ---
print("\n[evoked per-stimulus per-trial CV_P]")
for s in sorted(set(evk_stim.tolist())):
    sel = (evk_stim == s)
    R_s = evk_rates[:, sel]
    mr  = R_s.mean(axis=0)
    cvP = R_s.std(axis=0) / mr
    print(f"  stim={s}: n={int(sel.sum())} trials, mean_FR={mr.mean():.2f} Hz, "
          f"CV_P median={np.median(cvP):.4f}, S_P median="
          f"{np.median((1 - mr**2/((R_s**2).mean(axis=0)))/(1-1/N_V1)):.4f}")


# --- side-by-side direction summary ---
print("\n=== DIRECTION CHECK ===")
print(f"  per-trial CV_P:  spon median {np.median(cv_P_spon):.4f}  vs  evk median {np.median(cv_P_evk):.4f}  "
      f"-> {'evk > spon' if np.median(cv_P_evk) > np.median(cv_P_spon) else 'spon > evk'}")
print(f"  per-trial S_P :  spon median {np.median(S_P_spon):.4f}  vs  evk median {np.median(S_P_evk):.4f}  "
      f"-> {'evk > spon' if np.median(S_P_evk) > np.median(S_P_spon) else 'spon > evk'}")
print(f"  per-neuron CV_L: spon median {np.nanmedian(cv_L_spon):.4f}  vs  evk median {np.nanmedian(cv_L_evk):.4f}  "
      f"-> {'evk > spon' if np.nanmedian(cv_L_evk) > np.nanmedian(cv_L_spon) else 'spon > evk'}")
print(f"  per-neuron S_L : spon median {np.nanmedian(S_L_spon):.4f}  vs  evk median {np.nanmedian(S_L_evk):.4f}  "
      f"-> {'evk > spon' if np.nanmedian(S_L_evk) > np.nanmedian(S_L_spon) else 'spon > evk'}")
print(f"  per-trial mean: spon median {np.median(mean_spon):.3f} Hz  vs  evk median {np.median(mean_evk):.3f} Hz  "
      f"-> {'evk > spon' if np.median(mean_evk) > np.median(mean_spon) else 'spon > evk'}")
