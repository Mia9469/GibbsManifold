# Fig 3 — Single-neuron mutual-information rates

Generates **manuscript Fig. 3** (single-neuron LIF model: information
rate, ICE, and CV_L during training; ICE–CV_L plane with Gaussian-
channel reference).  Mutual information is computed by the Direct
Method (Strong et al. 1998) on spike words extracted from a frozen-
stimulus paradigm.  The single-neuron column of **manuscript Fig. 5**
(`figure5_singleneuron.fig`) is exported by these scripts and consumed
in `Fig5/` for the cross-system overlay.

## Entry-point hierarchy

| Tier            | Script                                | Role                                                              |
| --------------- | ------------------------------------- | ----------------------------------------------------------------- |
| **MAIN**        | `figure3_main.m`                      | Canonical pipeline: LIF + iSTDP simulation → spike-word Direct-Method MI → writes `MI_results_direct.mat`.  Run this first. |
| Post-processing | `redo_MI_from_saved.m`                | Re-compute MI from `MI_results_direct.mat` with the cross-stim formulation (matches the V1 pipeline) and 1/T extrapolation diagnostics. |
| Diagnostic      | `check_R0_singleneuron.m`             | R₀ anchor sensitivity for the single-neuron column of Fig. 5A.   |
| Snippet         | `figure3_errbar_snippet.m`            | **NOT standalone.**  Code block (jackknife error bars + plotting) to splice into `figure3_main.m` from line 618 onwards if error-bar plots are wanted. |
| Dev / legacy    | `figure3_main_dev.m`                  | Earlier dev version of the Direct-Method pipeline; writes `MI_results_direct_1.mat`.  Kept for reproducibility but **superseded by** `figure3_main.m`. |
| Dev / legacy    | `figure3_MI_histogram_legacy.m`       | Original histogram-based MI version (pre-Direct-Method).  Kept for reference. |

## Output `.fig` panels in this folder

| File                  | Manuscript panel                          |
| --------------------- | ----------------------------------------- |
| `figure3_panels.fig`  | Composite of Fig. 3 panels (assembly aid) |
| `figure3_cv.fig`      | Fig. 3E — CV_L vs. time                   |
| `figure3_f.fig`       | Fig. 3 firing-rate panel                  |
| `figure3_t.fig`       | Fig. 3 current traces                     |

## Typical workflow

```bash
matlab -batch "figure3_main"            # canonical pipeline (simulation + Direct-Method MI)
matlab -batch "redo_MI_from_saved"      # cross-stim formulation + 1/T diagnostic
matlab -batch "check_R0_singleneuron"   # R₀ sensitivity for Fig. 5A
```

## Method

Spike trains are binned at Δτ = 10 ms in 600-bin windows under a
frozen-stimulus paradigm (NStim = 20 templates × NReps = 10 repeats).
Binary words of length T ∈ {4, 5, 6, 7, 8, 10, 13, 15, 20, 25, 30, 40}
bins are formed and the entropy rate is extrapolated as

```
H∞ = lim_{T→∞} H(T) / (T · Δτ)     [bits/s]
```

via linear regression of `H(T)/T` against `1/T`.  Mutual information
is `I = H_total − H_noise`.  Cost `C = 2.4 + 0.37·r` (W) gives
`ICE = I/C` [bits/J].

Requires `compute_entropy_toolbox.m` on the MATLAB path.

## Outputs

* `MI_results_direct.mat` — per-T entropy estimates, bootstrap CIs, final `H∞`/`I∞` summaries.
* `MI_results_direct_1.mat` — output of `figure3_main_dev.m` (parallel/dev branch).
* `.fig` panels listed above.
* Single-neuron `(C, I, H, ICE)` summary exported and consumed by `Fig5/` for the cross-system overlay.

## Dependencies

MATLAB R2022a+ (Statistics and Machine Learning Toolbox) + `compute_entropy_toolbox.m`.
