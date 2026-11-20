# Fig 4 — Recurrent E–I spiking network with plasticity

Generates **manuscript Fig. 4**: a 300-E + 75-I spiking network in which
E–I plasticity changes reconstruction, current balance, firing-rate
statistics, population dispersion, estimated information, and raster
similarity. The complete 29-checkpoint weight archive is available in
`checkpoint.mat`. The corrected information workflow replays the same five
frozen stimuli at every checkpoint with 20 independent test repeats. Main-text
panel J currently reports mean neuron-wise matched-word MI divided by mean-unit
cost. The population decoder's Fano bound is a separate Supplementary analysis.
The move from historical pooled words at 0.1 ms to neuron-wise words at 1 ms
changes the readout and protocol, not merely an entropy implementation bug.
See `../outputs/network_MI_change_explanation_20260907/MI_changes_CN.md` for
the controlled comparison and exact repair-only reproduction of the old endpoint.
All analyses follow one learned trajectory, so repeats quantify test-noise or
resampling stability rather than independent training replication.

## Entry-point hierarchy

| Tier            | Script                          | Role                                                            |
| --------------- | ------------------------------- | --------------------------------------------------------------- |
| **TRAIN**       | `train_network.m`               | Driver that runs the network learning (calls `Learning.m`).  Sets `NneuronE = 300`, `NneuronI = 75`, `Nx = 3`, `lambda = 50`, learning rates `epsf`, `epsr`, etc.  Writes `checkpoint.mat`. |
|   ↳ uses        | `Learning.m`                    | Function: actual integrate-and-fire E/I network with iSTDP / Hebbian / voltage-STDP plasticity rules. |
|   ↳ uses        | `runnet.m`                      | Function: forward-pass network simulation (no learning), called for evaluation phases. |
| **INFORMATION** | `recompute_network_information.py` | Replays all checkpoints, writes packed 1-ms rasters, estimates neuron-wise $I((S,J);W)$ with matched sampling, performs leave-repeat-out population decoding, computes the Fano lower bound, and runs window/repeat/null audits. |
| Current figure assembly | `../scripts/export_mi_revision_panels.py` | Exports current word-MI panel J and review composites. Does not overwrite author artwork. |
| Historical figure assembly | `recompose_with_corrected_information.py` | Earlier decoder-primary layout; not the current main-text J. |
| Historical analysis | `figure4_main.m`          | Legacy checkpoint analysis retained for provenance only. Its information branch mixed neuron identity with repeated trials, changed the frozen stimulus across checkpoints, and constructed words at the 0.1-ms integration step. |
| **RELIABILITY** | `reliability.m`                 | Trial-to-trial response consistency for current Fig. 4K. Computes pairwise SSIM and `corr2` between spike rasters across repeated trials (`recorded_spikes_*_rec.mat`), and within-trial neuron-pair correlation similarity (`pattern_rep`). SSIM is distinct from the Poisson count-agreement metric $P_{agree}$ in Fig. 4L. |
| Uploaded-trial K reanalysis | `../scripts/recompute_figure4k_uploaded_trials.py` | Uses 50 uploaded endpoint trials per state and 25 disjoint pairs for an exact rank-sum test. Exports a separate K, not yet incorporated into manuscripts. |
| **BAR PLOTS**   | `fig3plot.m`                    | Historical before-vs-after plotter. Its legacy MI/ICE branch is not used; unaffected dispersion summaries remain provenance for the current figure. *Name retained from an earlier figure order.* |
| Historical post-processing | `recompose_without_uncorrected_information.py` | Archived interim step that removed the invalid legacy panel before the corrected checkpoint replay was completed. Not used for the current manuscript. |
| Diagnostic      | `Learning_testentr.m`           | Historical entropy-rate evaluation; not used for a current result. |
| Snippet         | `figure4_errbar_snippet.m`      | **NOT standalone.** Historical checkpoint-information plotting block. |
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
matlab -batch "reliability"            # repeated-trial raster-similarity analysis
python3 recompute_network_information.py
python3 recompute_network_information.py --reuse-rasters  # rerun estimators without replaying the network
python3 ../scripts/export_mi_revision_panels.py
```

## Model

* 300 excitatory + 75 inhibitory spiking neurons; 3-dimensional input.
* Voltage-based STDP on recurrent weights Ω; Hebbian rule with decay
  on feedforward weights F (Methods §3.6).
* Full parameter list in Supp Table S8.

## Outputs (most are NOT tracked by git; `.gitignore` excludes the large `.mat`)

* `checkpoint.mat` (≈ 122 MB) — complete 29-checkpoint weights and model state used by the corrected replay.
* `network_information_corrected/network_information_rasters_packbits.npz` — packed 1-ms excitatory spike rasters for five stimuli × 20 repeats × 29 checkpoints.
* `network_information_corrected/network_information_results.npz` and `network_information_summary.json` — corrected Direct-Method, decoder, cost, fit-quality, sensitivity, and null results.
* `../outputs/mi_revision_0905_20260907/panels/Figure4_J.pdf` — current main-text word-MI panel J.
* `network_information_corrected/Figure4J_network_information_corrected.pdf` — earlier decoder-primary panel, retained for provenance.
* `network_information_corrected/network_information_audit.pdf` — six-panel Supplementary audit.
* `learned_o_2025_20.mat` — non-information cache: 29 learned-weight checkpoints, 50 stochastic response realizations for population dispersion, and 10 SSIM resampling batches.
* `figure4_state.mat`, `Info_all_100.mat`, `500_rec.mat` — historical/intermediate snapshots; their legacy information fields do not underlie a current claim.
* `Figure4_corrected_with_withheld_panel.pdf` — A–L layout archive used as the base for corrected panel insertion.

## Original-protocol replay (separate review output)

`python3 scripts/replay_network_original_protocol.py` (from the repository root)
retests all 29 original weight checkpoints with one frozen input per checkpoint,
10 repeats and 0.1-ms pooled words. Every new raster is scored by the legacy
estimator and the two entropy repairs. Complete rasters, inputs and seeds are
saved in `outputs/network_original_protocol_replay_20260908/`.

This is new test data, not recovery of the old random realization. The repaired
replay MI/cost goes from 0.2842 to 0.8389; the historical final-checkpoint-only
repair of 0.7152 to 0.8766 remains separate. Neither curve is spliced into the
other. The two vector review PDFs in `output/pdf/` are not integrated into the
manuscript. See the replay directory's `README_CN.md` for scope and validation.
Run `python3 scripts/test_network_original_protocol_replay.py` for focused tests.

## Dependencies

* MATLAB R2022a+ (Statistics and Machine Learning Toolbox) for training and historical analyses
* Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, Numba 0.60.0, Matplotlib, and PyMuPDF for corrected replay and figure assembly
