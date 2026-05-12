# Fig 3 — Single-neuron mutual-information rates

Generates **manuscript Fig. 3** (single-neuron LIF model: information
rate, ICE, and CV_L during training; ICE–CV_L plane with Gaussian-
channel reference).  Mutual information is computed by the Direct
Method (Strong et al. 1998) on spike words extracted from a frozen-
stimulus paradigm.  The single-neuron column of **manuscript Fig. 5**
(`figure5_singleneuron.fig`) is exported by these scripts and consumed
in `Fig5/` for the cross-system overlay.

## Entry points

| Script                              | Produces                                                |
| ----------------------------------- | ------------------------------------------------------- |
| `figure3_main.m`                    | **Main pipeline** — runs the LIF model and writes per-time-point ICE / H / I / CV_L / cost trajectories |
| `figure3_MI_new.m`                  | Direct-Method mutual-information rate (current version)|
| `figure3_MI.m`                      | Earlier MI version, retained for reference              |
| `figure3_MI_plotting_errbar.m`      | Bootstrap error-bar plotting on MI vs. time             |
| `check_R0_singleneuron.m`           | R₀ sensitivity diagnostic for the per-system anchor used in Fig. 5A |
| `redo_MI_from_saved.m`              | Re-plot panels from saved per-T entropy estimates       |

Output panel `.fig` files in this folder:

| File                  | Used in                                            |
| --------------------- | -------------------------------------------------- |
| `figure3_panels.fig`  | Composite Fig. 3 panels (assembled in Illustrator) |
| `figure3_cv.fig`      | Fig. 3 — CV_L vs. time (panel E)                   |
| `figure3_f.fig`       | Fig. 3 — firing-rate panel                         |
| `figure3_t.fig`       | Fig. 3 — current traces                            |

Typical workflow:

```bash
matlab -batch "figure3_main"            # full single-neuron simulation
matlab -batch "figure3_MI_new"          # MI by Direct Method
matlab -batch "figure3_MI_plotting_errbar"
```

## Method

Spike trains are binned at Δτ = 10 ms in 600-bin windows under a
frozen-stimulus paradigm (20 templates × 10 repeats).  Binary words
of length T ∈ {4, 5, 6, 7, 8, 10, 13, 15, 20, 25, 30, 40} bins are
formed and the entropy rate is extrapolated as

```
H∞ = lim_{T→∞} H(T) / (T · Δτ)     [bits/s]
```

via linear regression of `H(T)/T` against `1/T`.  Mutual information
is `I = H_total − H_noise`.

All simulation parameters are listed in main-text Methods §3.5 and
Supp Table S8 (membrane time constant τ = 20 ms, refractory 5 ms,
iSTDP learning rate, target rate ρ₀ = 5 Hz, …).

## Outputs

* `MI_results_direct.mat` — per-T entropy estimates, bootstrap CIs, and
  final `H∞` / `I∞` summaries.
* `.fig` panels listed above.
* Single-neuron summary (C, I, H, ICE arrays) exported and consumed
  by `Fig5/` for cross-system overlay.

## Dependencies

MATLAB R2022a+ (Statistics and Machine Learning Toolbox).
