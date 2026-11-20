# Reproducing the current numerical analyses

## Scope

The reference manuscript is the November 2026 version entitled *How learning
reshapes neural responses under a spike budget: A maximum-entropy benchmark*.
Current figure numbers are mapped in README.md. There is no new animal-data
collection in this release. Generated responses are explicitly identified as
simulations. The population analyses do not estimate a 75-dimensional joint
entropy or identify a universal biological efficiency law.

## 1. Core environment and integrity

Python 3.9, NumPy 2.0.2, SciPy 1.13.1, Numba 0.60.0 and Matplotlib 3.9.4 were
used for the core tests. Install `requirements-core.txt`; then run:

```sh
python scripts/verify_release.py
```

This checks the allowlisted release hashes, all 24 training cases and the
budget/readout contracts. It does not claim every external-data pipeline was
rerun in the release environment. Scientific source files are frozen; any
packaging-only changes are documented in the release verification report.

## 2. Budget-controlled learning: current Fig. 2

`outputs/snn_budget_frontier_20261016/replication_protocol.json` records 8 seeds
and 3 activity targets (240/480/720 spikes per second across 75 cells), 5 saved
stages, training duration, calibration/test templates and evaluation seeds.
Each case contains a JSON record, packed synchronous test spikes and saved
weights. The E+I budget is controlled during both training and evaluation.

```sh
python -m unittest discover -s outputs/snn_budget_frontier_20261016 -p 'test_*.py'
python outputs/snn_budget_frontier_20261016/summarize.py
python outputs/snn_budget_frontier_20261016/verify_experiment.py
```

To retrain rather than reuse the saved cases, use a separate copy of this
repository, rename that copy's `replication/` directory, and run:

```sh
python outputs/snn_budget_frontier_20261016/replicate_learning.py --workers 3
```

Do not mix this 60E/15I cohort with the original 300E/75I trajectory. The primary
one-second count support, achieved-mean reference and seed-level inference are
fixed in the protocol. A negative delta of the nonnegative entropy gap means
approaching the reference, not exceeding its entropy.

## 3. Cell identity, dependence, cell type and same-response readout

```sh
python outputs/elife_reconsideration_20261109/analysis/population_bridge.py --self-test
python outputs/elife_reconsideration_20261109/analysis/population_bridge.py
python outputs/elife_editor_focus_20261111/analysis.py
```

The first analysis uses cached, hash-bound computations when present; remove
only those per-case JSONs in a disposable copy to force the full shuffled
dependence reanalysis. The second reconstructs cell-type/window tables and the
two-cell count/readout comparison directly from the saved synchronous responses.
Joint-gap and reconstruction use the same 20-ms excitatory-pair count features;
that readout is different from the full excitatory filtered-spike readout.
Groups/windows/shuffles are averaged within each training seed, not treated as
independent trained networks. The E/I decomposition is statistical accounting,
not a causal E/I-allocation intervention.

## 4. One-second sampling and achieved-cost sensitivity

```sh
python outputs/elife_one_second_g_20261111/verify.py
python outputs/elife_one_second_g_20261111/sensitivity.py --with-replay
```

Saved resampling arrays and the 48 endpoint replay summaries are included.
`verify.py` independently checks input hashes, selected resampling draws and
the matched-cost entropy identity. It does not itself execute every stochastic
replay. To execute frozen-weight endpoint replays use `replay_cost.py`; this
does not retrain the networks. Resampling robustness of the one-second pooled
result does not establish its extension to other windows or cell selections.

## 5. V1 and cross-system recordings: current Figs. 3-4

CRCNS V1/V2 data: <https://crcns.org/data-sets/vc/v1v2-1>. Obtain registration
and observe the provider's terms. Install `requirements-recordings.txt` for
the Python V1 analyses. Place these files in `Fig2/`:

```text
105l001p16.mat
106r001p26.mat
106r002p70.mat
107l002p67.mat
107l003p143.mat
```

```sh
python Fig2/v1_decoder_crossfit.py
python Fig5/discriminative_maxent_test.py
python Fig5/shared_shape_count_loso.py
python Fig2/v1_mixedeffects_state.py
```

The raw-data-dependent paths are not advertised as a no-input smoke test.
Saved JSON/CSV summaries are provided separately from raw neural recordings.
MATLAB scripts require R2022a or compatible. Word-based estimates retain zero-
entropy positions and match total/conditional sampling starts; these checks
do not remove the finite-repeat limitations described in the manuscript.

Additional raw inputs: [Allen Visual Coding Neuropixels](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels),
[Steinmetz](https://osf.io/agvxh/), and [IBL Brain-Wide Map](https://registry.opendata.aws/ibl-brain-wide-map/).
Provider-specific access and optional dependencies are documented by the Fig2
entry points. Do not redistribute provider recordings based on the code license.

## 6. Model information readouts and single-neuron trajectory

The original common-input network responses have shape 29 checkpoints x 5
templates x 20 repeats x 300 excitatory neurons x 1000 binary 1-ms bins.

```sh
python scripts/recompute_network_words.py \
  --rasters Fig4/network_information_corrected/network_information_rasters_packbits.npz \
  --reference Fig4/network_information_corrected/network_information_results.npz \
  --stages 1 29
```

Omit `--stages` for all 29 checkpoints; add `--all-word-lengths` for non-fitted
word-length diagnostics. This independently rebuilds matched sliding words,
both entropy terms and the estimated per-spike curve. It writes under
`recomputed/`, not the submitted figure/PDF directories. This is the Appendix
original-network readout, not the new budget-controlled experiment.

`Fig3/` contains the single-neuron MATLAB simulation and corrected-information
analysis. `Fig3/MI_results_direct_corrected.mat` and `source_data/` contain
saved author-generated model results; these are not independent retraining
replications. The full 20-window trajectory retains the initial transient.

## 7. Sparse coding: current Fig. 6

Install `requirements-ann.txt`. Obtain the original whitened `IMAGES.mat` from
the dataset distribution referenced by <https://github.com/hunse/vanhateren>
and place it at `Fig6/src/model/IMAGES.mat`. Image source, whitening and array
identity matter; a different image bank is not an exact frozen-input replay.

The current CVM constraint is `cv_constrained_v2`; compare Null, SparseNet and
CVM under common inference. Training/reconstruction code is in `Fig6/train_new.py`,
`generalization_test.py` and `heldout_generalization_test.py`. The training and
evaluation commands can generate fresh checkpoints; `--skip-train` requires the
corresponding saved checkpoint files, not merely the released JSON summaries.

```sh
python -m unittest discover -s Fig6/tests -p 'test_*.py'
python Fig6/generalization_test.py --n-runs 20 --n-epochs 10
python Fig6/heldout_generalization_test.py --cv loo --n-runs 6 --n-epochs 10
```

The frozen run/fold/pattern summaries and exact structured input bank are
included. Natural-image inference uses image folds; structured comparisons
retain pattern-level paired summaries and initialization variability. The
legacy output key `ICE` denotes the dimensionless reconstruction-activation
score (RAE), not measured mutual information. Complete historical pickled
training objects are not part of the minimal code release; do not claim that
JSON verification alone re-executes their training. Image reconstruction error
is not one minus classification accuracy.

## Historical files and limits

`legacy_202605/` retains pre-revision source snapshots. They do not define
current figure numbers, claims, estimator conventions or journal status.
The current submitted manuscript, appendix and signed letter are deliberately
not republished or modified by this code-release operation. An integrity test
can verify a saved result without resolving its sampling bias or establishing
a broader optimality mechanism.
