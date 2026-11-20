# Fig 3 — Single-neuron mutual-information rates

Generates **manuscript Fig. 3** (single-neuron LIF model: corrected
fine-time information, information per calibrated cost proxy, and CV_L
during training). Mutual information is computed by the Direct Method
on spike words extracted from a frozen-stimulus paradigm. The corrected
cache is consumed by `Fig5/make_corrected_figure5.py`.

## Entry-point hierarchy

| Tier            | Script                                | Role                                                              |
| --------------- | ------------------------------------- | ----------------------------------------------------------------- |
| **MAIN**        | `figure3_main.m`                      | Canonical pipeline: LIF + iSTDP simulation → matched-sampling spike-word Direct-Method MI → writes `MI_results_direct_corrected.mat`. |
| Post-processing | `redo_MI_from_saved.m`                | Re-compute corrected MI from the rasters in the legacy cache, leave that input unchanged, and write `MI_results_direct_corrected.mat`. |
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
Binary words of length T from 4 to 100 bins are formed, and the retained
20--100-bin range is used for the finite-word extrapolation

```
H∞ = lim_{T→∞} H(T) / (T · Δτ)     [bits/s]
```

via linear regression of `H(T)/T` against `1/T`. Total and conditional
entropy use the same sliding word starts, and valid zero-entropy
conditional positions remain in the average. The resulting estimand is
`I((S,J);W) = H(W) - H(W|S,J)`, where `J` is relative word-start time;
it is not identity-only `I(S;W)`. The calibrated affine proxy
`C̃ = 2.4 + 0.37·r` gives the model-defined ratio `I/C̃`.  The
coefficients are not per-neuron power measurements, so the ratio is
not reported as absolute bits/J.

Requires `compute_entropy_toolbox.m` on the MATLAB path.

## Outputs

* `MI_results_direct_corrected.mat` — corrected per-T curves and final information summaries used in the manuscript.
* `MI_results_direct.mat` — legacy raster/cache input retained for provenance; its saved information fields are not used.
* `MI_results_direct_1.mat` — output of `figure3_main_dev.m` (parallel/dev branch).
* `.fig` panels listed above.
* Single-neuron `(cost proxy, I((S,J);W), H, CV)` summary consumed by `Fig5/` for the corrected projection.

## Dependencies

MATLAB R2022a+ (Statistics and Machine Learning Toolbox) + `compute_entropy_toolbox.m`.
