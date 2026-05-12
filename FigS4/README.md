# Supp Fig S4 — Single-neuron LIF model with inhibitory STDP

Generates **Supplementary Fig. S4** (formerly main-text Fig. 3, demoted
to supplementary in the latest manuscript version): a conductance-based
leaky integrate-and-fire (LIF) neuron receiving eight excitatory +
inhibitory input channels, with inhibitory STDP driving E–I balance.
Tracks ICE and CV_L along learning and reveals that CV_L saturates
while ICE keeps climbing (proxy-saturation limit).

This folder is kept distinct from `Fig3/` (which now hosts the
single-neuron mutual-information rates that are the new main-text
Fig. 3); the two single-neuron analyses share the model but address
different questions.

## Entry points

| Script                  | Produces                                                  |
| ----------------------- | --------------------------------------------------------- |
| `figure3.m`             | Core simulation: current traces, separability matrix, ICE / CV_L over time |
| `figure3_HC_ICE.m`      | ICE / H–C trajectory analysis (assembles panels E–G)      |
| `figure3_panelD.m`      | Per-channel current profiles + stimulus-separability matrix Ū |
| `figure_HC_phase.m`     | Phase-plane visualization on the ICE–CV plane             |

Typical workflow:

```bash
matlab -batch "figure3"
matlab -batch "figure3_HC_ICE"
matlab -batch "figure3_panelD"
```

## Parameters

All simulation parameters are listed in Supp Table S8 (membrane time
constant τ = 20 ms, refractory 5 ms, iSTDP learning rate η = 0.0012,
target rate ρ₀ = 5 Hz, …).  A frozen-stimulus paradigm (20 templates ×
10 repeats) is used for mutual-information evaluation.

## Outputs

Outputs are saved as MATLAB `.fig` figures and `.mat` per-time-point
state snapshots.  Final paper PDF (`FigureS4.pdf`) is assembled in
`updated/figures/`.

## Dependencies

MATLAB R2022a+ (Statistics and Machine Learning Toolbox).
