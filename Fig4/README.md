# Fig 4 — Recurrent E–I spiking network with plasticity

Generates **manuscript Fig. 4**: a 300-E + 75-I spiking network that
self-organises through E–I plasticity into a high-CV, high-ICE regime.
Reconstruction error decreases while population CV / sparseness rise;
ICE and mutual-information rate both increase during training.  The
network column of **manuscript Fig. 5** (`figure5_network.fig`) —
restricted to the spatiotemporal operating points — is exported here
and consumed in `Fig5/`.

## Entry-point hierarchy

| Tier            | Script                          | Role                                                            |
| --------------- | ------------------------------- | --------------------------------------------------------------- |
| **TRAIN**       | `train_network.m`               | Driver that runs the network learning (calls `Learning.m`).  Sets `NneuronE = 300`, `NneuronI = 75`, `Nx = 3`, `lambda = 50`, learning rates `epsf`, `epsr`, etc.  Writes `checkpoint.mat`. |
|   ↳ uses        | `Learning.m`                    | Function: actual integrate-and-fire E/I network with iSTDP / Hebbian / voltage-STDP plasticity rules. |
|   ↳ uses        | `runnet.m`                      | Function: forward-pass network simulation (no learning), called for evaluation phases. |
| **ANALYSIS**    | `figure4_main.m`                | Canonical Direct-Method MI pipeline on the trained network (loads `checkpoint.mat`, frozen-stimulus paradigm, Trials × Stages outer loops).  Produces Fig. 4 panels. |
| Post-processing | `redo_fig4_plots.m`             | Apply Cost-unit fix in place and redraw Fig. 4 panels (skips simulation). |
| Diagnostic      | `Learning_testentr.m`           | Entropy-rate evaluation along the learning trajectory (older companion to `figure4_main.m`). |
| Snippet         | `figure4_errbar_snippet.m`      | **NOT standalone.**  PART A (per-neuron MI storage) + PART B (plot replacement) to splice into `figure4_main.m`. |
| Legacy          | `figure4_legacy.m`              | Pre-Direct-Method version (histogram MI).  Kept for reference.  |
| Legacy          | `Fano_USE.m`                    | Decoding-error + rate evolution analysis (March 2025).  Legacy reliability proxy. |
| Legacy          | `FanoCV.m`                      | Fano-factor / CV computation on trained network.  Legacy.       |

## Output `.fig` panel in this folder

| File              | Used in                                  |
| ----------------- | ---------------------------------------- |
| `figure4_t.fig`   | Fig. 4 current/spike-time traces (or fed into Fig. 5B export) |

## Typical workflow

```bash
matlab -batch "train_network"          # full network training, writes checkpoint.mat (≈ 122 MB)
matlab -batch "figure4_main"           # Direct-Method MI analysis on the trained network
matlab -batch "redo_fig4_plots"        # quick re-plot with cost-unit fix (no re-sim needed)
```

## Model

* 300 excitatory + 75 inhibitory spiking neurons; 3-dimensional input.
* Voltage-based STDP on recurrent weights Ω; Hebbian rule with decay
  on feedforward weights F (Methods §3.6).
* Frozen-stimulus paradigm for MI evaluation: one stimulus per learning
  stage, replayed across multiple trials to estimate noise entropy from
  trial-to-trial variability.
* Full parameter list in Supp Table S9.

## Outputs (most are NOT tracked by git; `.gitignore` excludes the large `.mat`)

* `checkpoint.mat` (≈ 122 MB) — full simulation state used by `figure4_main.m`.
* `figure4_state.mat`, `learned_o_2025_20.mat`, `Info_all_100.mat`, `500_rec.mat` — intermediate state snapshots.
* `.fig` panels for Fig. 4 and the network column of Fig. 5.

## Dependencies

* MATLAB R2022a+ (Statistics and Machine Learning Toolbox)
* `compute_entropy_toolbox.m` on path (for Direct-Method MI)
