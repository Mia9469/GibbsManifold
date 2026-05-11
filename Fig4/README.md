# Fig 4 — Mutual-information rates (frozen-stimulus paradigm)

Computes mutual-information rates `I(s; r)` for the single-neuron and
recurrent-network simulations using the Direct Method with finite-T
extrapolation (Strong et al. 1998).  Provides the MI summaries that
feed into **manuscript Fig. 3J** (single-neuron) and **Fig. 4J**
(recurrent network), as well as the **R₀ calibration check** used in
Fig. 5.

## Entry points

| Script                            | Produces                                              |
| --------------------------------- | ----------------------------------------------------- |
| `figure4_MI_new.m`                | Main MI computation pipeline (current version)        |
| `figure4_MI.m`                    | Earlier version, retained for reference               |
| `figure4_MI_plotting_errbar.m`    | Bootstrap error-bar plotting                          |
| `redo_MI_from_saved.m`            | Re-plot panels from saved per-T entropy estimates     |
| `check_R0_singleneuron.m`         | R₀ sensitivity diagnostic for single-neuron MI        |

Typical workflow:

```bash
matlab -batch "figure4_MI_new"
matlab -batch "figure4_MI_plotting_errbar"
```

## Method

Spike trains are binned at Δτ = 1 ms (recurrent) or 10 ms (single
neuron).  Binary words of length T ∈ {4, 5, 6, 7, 8, 10, 13, 15, 20,
25, 30, 40} ms are formed and the entropy rate is extrapolated as

```
H∞ = lim_{T→∞} H(T) / (T·Δτ)     [bits/s]
```

via linear regression of `H(T)/T` against `1/T`.  Mutual information
is `I = H_total − H_noise`.

## Outputs

`.mat` files containing per-T entropy estimates, bootstrap confidence
intervals, and final `H_∞` / `I_∞` summaries.

## Dependencies

MATLAB R2022a+.
