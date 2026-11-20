# Sparse-coding module: CV-max proof of principle

This directory contains the sparse-coding engineering analysis relocated to
Supplementary Figs. S20-S21 in ANN-in-Supp (main Fig. 7 and Supplementary
Fig. S20 in ANN-in-Main), plus the matched-baseline ablation in Fig. S8.
It is an exploratory engineering comparison, not biological validation or a claim of
absolute energy efficiency.

## Current algorithm

All manuscript analyses use `cv_constrained_v2`, the absolute-RE-target gate.
The earlier `cv` and warmup-relative `cv_constrained` modes remain only for
legacy sensitivity checks and must not be mixed into current aggregates.

The models use 144 dictionary atoms for 16 x 16 (256-dimensional) patches, so
the dictionary is **undercomplete**, not overcomplete. All methods share the
same ISTA soft threshold (`lambda=5e-3`):

1. **Null**: shared L1/ISTA inference, with no additional Cauchy or CV term.
2. **SparseNet**: shared L1 inference plus a Cauchy penalty.
3. **CV-max-v2**: shared L1 inference plus an adaptively scaled population-CV
   term, active only after relative reconstruction error reaches the target
   (`rho=0.40`, target `RE=0.60` in the headline configuration).

The dimensionless engineering score is
`RAE = -log2(RE) / mean_abs_activation` under the common activation
normalization. It is neither measured mutual information nor
hardware/metabolic energy. Some archived CSV/JSON fields retain the legacy key
`ICE` for backward compatibility; manuscript labels and new outputs use RAE.

## Reproduction

For Supplementary Fig. S6, run `python3 Fig6/make_figureS6_v2.py` from the
repository root. It reads all 600 epoch-10 checkpoints from the frozen headline
cohort, validates configuration and finite coefficients, and checks their run
summaries against the final training records. All three panels are endpoint
metrics. Earlier S6 caption CV values were averages over all ten training epochs
in `result_new.py`, not final-checkpoint CV; that legacy JSON remains unchanged.
Outputs include the vector figure, all 60 run-level rows, corrected pairwise
tests, and checkpoint SHA-256 hashes in `outputs/supp_revision_20260904/S6/`.

Run from `Fig6/`:

```bash
python3 train_new.py --cons non --n_runs 20 --epoch 10 --seed-base 20260630
python3 train_new.py --cons s --n_runs 20 --epoch 10 --seed-base 20260630
python3 train_new.py --cons cv_constrained_v2 --recon-budget 0.60 \
  --cv_ratio 0.40 --reg 5e-3 --n_runs 20 --epoch 10 \
  --seed-base 20260630

# Set FIG6_WRITE_FIGURES=0 for statistics-only verification.
FIG6_WRITE_FIGURES=0 python3 result_new.py

# Matched-baseline sweep: Null 24 + Cauchy 24 + CVM-v2 48 = 96 runs.
bash p3_matched/p3_sweep.sh
python3 p3_matched/p3_eval.py --methods null,cauchy,cvmax_v2
python3 p3_matched/plot_figureS8.py
```

`train_new.py` uses seed `seed_base + run_index` and stores the seed and full
headline configuration in each `.mat` record. Existing figure-source records
created before this change are valid stochastic runs but do not contain
recoverable seed metadata. We therefore performed a clean deterministic audit
of all three methods with the same 20-seed panel (`20260630`--`20260649`). Every
metric ranking was reproduced; the largest group-mean shift was 0.0227
percentage points for CVM dictionary independence, and all corrected pairwise
tests remained below `1.5e-7`. Compact results are in
`seeded_rerun_validation.json`. The archival values are retained as the figure
source so that the audit does not silently change the submitted panels.

## Entry points

| Script | Role |
| --- | --- |
| `train_new.py` | Deterministic Null, SparseNet, and CVM-v2 training |
| `result_new.py` | Run-level evaluation; Welch tests with within-metric Bonferroni correction |
| `generalization_test.py` | Matched-inference natural and structured tests |
| `heldout_generalization_test.py` | Leave-one-image-out natural-image analysis |
| `gen_stats.py` | Unit-aware effect-size summaries |
| `analyze_models.py` | Optional legacy dictionary/noise diagnostics |
| `p3_matched/` | Current v2 matched-RE and matched-activation sweep |

## Outputs

- `record/`: run records and checkpoints (ignored by git).
- `sparse_run_statistics.json`: run-level group summaries and adjusted tests.
- `generalization_results/`: machine-readable natural/OOD summaries.
- `p3_matched/results/eval/`: 96-run v2 sweep tables.
- `updated/figures/Figure6_ann_combined_RAE.pdf`: combined training and
  generalization panels (main-text Fig. 7 in the ANN-in-main build;
  Supplementary Fig. S21 in the ANN-in-supplement build).
- `updated/figures/Figure6_generalization.pdf`: the two reconstruction-error
  controls retained in the Supplement (Fig. S21 or S22, depending on build).
- `updated/figures/Figure6.pdf`: superseded six-panel layout retained for
  provenance; it is no longer included by the current TeX sources.
- `updated/figures/FigureS8.pdf`: matched-baseline Pareto analysis.

## Dependencies and data

Python 3.10 or newer with NumPy, SciPy, pandas, Matplotlib, seaborn,
scikit-image, and PyTorch 2 or newer. The input is
`src/model/IMAGES.mat` (van Hateren natural images); CUDA is optional.
