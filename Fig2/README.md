# Fig 2 — Macaque V1/V2 Direct-Method analysis

Generates **main-text Fig. 2** (lifetime CV / sparseness separating
spontaneous and evoked V1 states), **Supp Fig. S3** (per-trial
population CV/SP), and **Supp Fig. S11** (FR-controlled per-neuron
decoder validation).

## Required data

Five session files from the CRCNS **`v1v2-1`** dataset
(<https://crcns.org/data-sets/vc/v1v2-1>) — download into this folder:

```
105l001p16.mat   106r001p26.mat   106r002p70.mat
107l002p67.mat   107l003p143.mat
```

These files are *not* redistributed here; they require a free CRCNS
account.

## Entry points

| Script                          | Produces                                       |
| ------------------------------- | ---------------------------------------------- |
| `fig2_stats.m`                  | Within-condition CV_L, S_L, mean-rate stats (CSV) |
| `fig2_styled_panels.py`         | Main Fig. 2C / 2D paper-style panels           |
| `fig2_CVL_panels.py`            | CV_L raw panels (intermediate)                 |
| `figS_CVP_panels.py`            | Supp Fig. S3 per-trial CV_P / S_P panels       |
| `v1_decoder_vs_CV_multi.py`     | Supp Fig. S11 5-session decoder analysis       |
| `recompute_cv_py.py`            | Stand-alone CV-from-rasters utility            |
| `check_trial_structure.m`       | Quick sanity check on each session's raster    |

Typical workflow:

```bash
matlab -batch "fig2_stats"              # writes fig2_stats.csv
python3 fig2_styled_panels.py            # writes Fig2_panels_CD_CVL_styled.pdf
python3 v1_decoder_vs_CV_multi.py        # writes v1_decoder_vs_CV_multi.{json,pdf,txt}
```

## Outputs

| File                                 | Used in                       |
| ------------------------------------ | ----------------------------- |
| `fig2_stats.csv`                     | Supp Table S6                 |
| `Fig2_panels_CD_CVL_styled.pdf`      | Main Fig. 2 (panels C, D)     |
| `FigS_CVP_styled.pdf`                | Supp Fig. S3                  |
| `v1_decoder_vs_CV_multi.pdf`         | Supp Fig. S11                 |
| `v1_decoder_vs_CV_multi.json`        | Numerical summary             |

## Dependencies

* MATLAB R2022a+ (Statistics and Machine Learning Toolbox)
* Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`, `scikit-learn`
* `scikit-learn ≥ 1.3` (LDA + StratifiedKFold for decoder analysis)
