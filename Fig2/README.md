# Fig 2 — Macaque V1/V2 Direct-Method analysis

Generates **main-text Fig. 2** (lifetime CV / sparseness separating
spontaneous and evoked V1 states), the corrected V1 fine-time
information analysis, **main Fig. 6D-F / Supp Fig. S11**
(independent-half per-neuron decoder decomposition), and the IBL/BWM
state controls **Supp Figs. S15--S16**.

## Required data

Five session files from the CRCNS **`v1v2-1`** dataset
(<https://crcns.org/data-sets/vc/v1v2-1>) — download into this folder:

```
105l001p16.mat   106r001p26.mat   106r002p70.mat
107l002p67.mat   107l003p143.mat
```

These files are *not* redistributed here; they require a free CRCNS
account.

Supp Figs. S15--S16 additionally use the IBL Brain-Wide Map operating
point table generated from public OpenAlyx/IBL data:

```
../IBL_data/ibl_metadata/gm_ops_pooled_v2.csv
../IBL_data/ibl_metadata/learning_state_transition_region_metrics.csv
```

Those tables are produced by the companion IBL pipeline in the sibling
`IBL_data` repository and are not part of the CRCNS download.

## Entry points

| Script                          | Produces                                       |
| ------------------------------- | ---------------------------------------------- |
| `fig2_stats.m`                  | Within-condition CV_L, S_L, mean-rate stats (CSV) |
| `fig2_styled_panels.py`         | Main Fig. 2C / 2D paper-style panels           |
| `fig2_CVL_panels.py`            | CV_L raw panels (intermediate)                 |
| `figS_CVP_panels.py`            | Supp Fig. S3 pooled five-session CV_P / S_P panels |
| `v1_decoder_crossfit.py`        | Main Fig. 6D-F / Supp Fig. S11 independent-half decoder analysis and session-level inference |
| `v1_decoder_vs_CV_multi.py`     | Historical same-trial decoder analysis retained only for provenance |
| `v1_decoder_session_stats.py`   | Historical same-trial session summary retained only for provenance |
| `v1_mixedeffects_state.py`      | Five-session hierarchical state inference      |
| `v1_ice_cvl_session_stats.py`   | Five-session firing-rate-controlled I/C and H/C inference |
| `unified_v1_Hrho_ICE.m`         | Corrected V1 `I((S,J);W)`, entropy fractions, and proxy-cost ratios |
| `compute_v1_spon_Hr.m`          | Corrected spontaneous response-entropy cache for Fig. 5 |
| `make_entropy_sampling_schematic.py` | Supp Fig. S5 matched-sampling estimator schematic |
| `figure2E_cross_dataset_CVL.py` | Supp Fig. S12 multi-species / multi-region CV_L state contrast |
| `figure2E_CVP_sanity_check.py`  | CV_P companion check for Fig. 2E interpretation |
| `ibl_recruitment_vs_dCV.py`     | Supp Fig. S16: firing-rate and independent-decoder controls for the regional CV_L shift |
| `ibl_bwm_state_transition_figure.py` | Supp Fig. S15 IBL/BWM CV and sparseness state transition |
| `recompute_cv_py.py`            | Stand-alone CV-from-rasters utility            |
| `check_trial_structure.m`       | Quick sanity check on each session's raster    |

Typical workflow:

```bash
matlab -batch "fig2_stats"              # writes fig2_stats.csv
matlab -batch "unified_v1_Hrho_ICE"     # corrected matched-sampling information estimates
matlab -batch "compute_v1_spon_Hr"      # corrected spontaneous entropy estimates
python3 fig2_styled_panels.py            # writes Fig2_panels_CD_CVL_styled.pdf
python3 v1_decoder_crossfit.py           # writes v1_decoder_crossfit.{json,npz} and Figure_decoder_crossfit.pdf
python3 v1_mixedeffects_state.py         # writes v1_mixedeffects_state.json
python3 v1_ice_cvl_session_stats.py      # writes v1_ice_cvl_session_stats.json
python3 figure2E_cross_dataset_CVL.py
python3 figure2E_CVP_sanity_check.py
python3 ibl_bwm_state_transition_figure.py
```

## Outputs

| File                                 | Used in                       |
| ------------------------------------ | ----------------------------- |
| `fig2_stats.csv`                     | Supp Table S5                 |
| `Fig2_panels_CD_CVL_styled.pdf`      | Main Fig. 2 (panels C, D)     |
| `FigureS3.pdf`                       | Supp Fig. S3                  |
| `figS_CVP_summary.json`              | Numerical summary for S3      |
| `Figure_decoder_crossfit.pdf`        | Main Fig. 6D-F / Supp Fig. S11 |
| `v1_decoder_crossfit.npz`            | Per-neuron independent-half plotting values |
| `v1_decoder_crossfit.json`           | Five-session inference, animal-group directions and incremental rank-R2 |
| `v1_mixedeffects_state.json`         | Five-session state mixed model and exact sensitivities |
| `unified_v1_Hrho_ICE.mat`            | Corrected five-session fine-time information and entropy estimates |
| `spon_Hr_perN.mat`                   | Corrected spontaneous response entropy used in Fig. 5 |
| `v1_ice_cvl_session_stats.json`      | Five-session I/C and H/C rank-residual inference |
| `Figure2E_cross_dataset_CVL.pdf`     | Supp Fig. S12 multi-species / multi-region state contrast |
| `figure2E_cross_dataset_CVL_summary.json` | Numerical summary for Supp Fig. S12 |
| `Figure2E_CVP_sanity_check.pdf`      | Companion CV_P check          |
| `figure2E_CVP_sanity_check_summary.json` | Numerical summary for CV_P check |
| `FigureS18_recruitment_vs_dCV.pdf`   | Supp Fig. S16 (historical filename retained) |
| `ibl_recruitment_vs_dCV_summary.json` | Numerical summary for S16 |
| `FigureS16_IBL_BWM_state_transition.pdf` | Supp Fig. S15 (historical filename retained) |
| `ibl_bwm_state_transition_summary.json` | Numerical summary for S15 |

## Dependencies

* MATLAB R2022a+ (Statistics and Machine Learning Toolbox)
* Python 3.9.6 core environment from `../environment.yml`
* Allen Visual Coding download, when needed, from the isolated
  `../environment-allen.yml`
