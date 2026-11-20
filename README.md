# GibbsManifold

Code and shared numerical inputs for **How learning reshapes neural responses
under a spike budget: A maximum-entropy benchmark**.

Mingyi Huang, Lingyan Mao, Jing Ding, Anna Wang Roe, Wei Lin and Yuguo Yu.
This is a research-code release, not a statement of journal acceptance.

The current release distinguishes a cost-constrained response-entropy reference
from task performance. It includes the controlled 60-excitatory/15-inhibitory
network cohort, saved synchronous simulated responses and weights, population
and same-response readout analyses, and the accompanying V1 and sparse-coding
analysis code. A larger CV is not asserted to imply higher coding efficiency.

## Start here

Use Python 3.9 with the pinned core dependencies:

```sh
python -m pip install -r requirements-core.txt
python scripts/verify_release.py
python -m unittest discover -s outputs/snn_budget_frontier_20261016 -p 'test_*.py'
python outputs/snn_budget_frontier_20261016/summarize.py
python outputs/snn_budget_frontier_20261016/verify_experiment.py
python outputs/elife_reconsideration_20261109/analysis/population_bridge.py --self-test
python outputs/elife_one_second_g_20261111/verify.py
```

The first command verifies the released file hashes and key numerical contracts.
`verify_experiment.py` recomputes all 120 archived network checkpoints and replays
one initial/learned pair from the saved weights; it does not retrain the cohort.
Runtime and exact stochastic replay can vary by platform. The saved responses
permit direct statistical checks independently of stochastic replay.

## Current manuscript map

The historical `Fig*/` directory names are retained for import compatibility.
They are **not** the current article's figure numbers.

| Current result | Code and shared inputs |
| --- | --- |
| Fig. 1: specified entropy reference and conditional CV/sparseness relation | `outputs/snn_budget_frontier_20261016/frontier.py`, `test_frontier.py`; `Fig1/` contains historical illustrative calculations |
| Fig. 2: budget-controlled learning and pooled/joint response analyses | `outputs/snn_budget_frontier_20261016/`, `outputs/elife_editor_focus_20261111/analysis.py`, `outputs/elife_readout_review_20261111/figure2g_seed_deltas.csv` |
| Fig. 3: V1 response statistics and state comparisons | `Fig2/`, including the session-level and mixed-effects analyses |
| Fig. 4: response-law comparisons and independent-half decoding | `Fig5/discriminative_maxent_test.py`, `Fig5/shared_shape_count_loso.py`, `Fig2/v1_decoder_crossfit.py` |
| Fig. 5: single-neuron adaptive trajectory | `Fig3/`; `source_data/LIF_trajectory.csv` and the saved model analysis arrays |
| Fig. 6: sparse coding and held-out/structured generalization | `Fig6/`, including v2 CV targeting, finite-value guards, reference-range SSIM, run/fold/pattern summaries and tests |
| Appendix population diagnostics | `outputs/elife_reconsideration_20261109/analysis/` |
| Appendix one-second sampling and achieved-cost controls | `outputs/elife_one_second_g_20261111/` |
| Appendix original 300E/75I network information readout | `Fig4/network_information_corrected/`, `scripts/recompute_network_words.py` |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the analysis sequence, required
external inputs and the distinction between recomputation and retraining.
`RELEASE_MANIFEST.json` records hashes of the new release payload. Transport-only
redaction replaces the original author's absolute workspace prefix in JSON
metadata with repository-relative paths; numerical arrays and values are not
changed.

## Data and access

The packed rasters in the controlled-network and common-input network directories
are **author-generated simulations**, not animal recordings. The five CRCNS
session files, Allen/Steinmetz/IBL raw recordings and van Hateren images are not
redistributed. Obtain these from their original providers under their access
terms; the exact source links and expected paths are in REPRODUCIBILITY.md.

Historical manuscript snapshots and the old source-data workbook are retained
under `legacy_202605/` for provenance, not as the current submitted manuscript or
current source-data authority. Historical code that uses an entropy or cost
definition superseded by this release must not be substituted for the mapped
analyses above. This release does not modify or upload the submitted PDFs.

## License and citation

Source code retains the repository's MIT license. External datasets retain their
providers' terms. `CITATION.cff` supplies software and manuscript metadata; no
unissued DOI or journal acceptance is claimed.

Analysis contact: Mingyi Huang, `myhuang20@fudan.edu.cn`.
