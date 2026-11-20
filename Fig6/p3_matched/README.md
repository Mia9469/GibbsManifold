# P3 — Matched-baseline analysis for Fig. 6

This subdirectory implements the **matched-RE / matched-activation baseline**
that the reviewer asked for as the P3 priority item.  It compares three
sparse-coding training regimes on the natural-image patch dataset, holding
every architectural and training hyper-parameter equal except for the
regularisation knob:

| method     | objective              | SparseNet `cons` mode      | swept knob             |
|------------|------------------------|----------------------------|------------------------|
| `null`     | reconstruction + L1    | `non`                      | L1 strength `lmda` (8 levels) |
| `cauchy`   | reconstruction + L1 + Cauchy prior | `s`            | Cauchy weight `alpha_s` (8 levels) |
| `cvmax_v2` | reconstruction + L1 + **absolute-RE-target gated CV penalty** | **`cv_constrained_v2`** | 2-D grid: `recon_budget` (4 target REs: 0.4, 0.5, 0.55, 0.6) × `cv_ratio` (4) |

The manuscript uses only `cvmax_v2`, matching `USE_V2=True` in
[`Fig6/result_new.py`](../result_new.py). The warmup-relative `cvmax` v1 mode
is retained in the code only as a legacy sensitivity and is excluded from the
default sweep and aggregate tables.

Sweep totals:
- null:     8 × 3 seeds        = 24 runs
- cauchy:   8 × 3 seeds        = 24 runs
- cvmax_v2: 4 × 4 × 3 seeds    = 48 runs
- **Total: 96 runs**, ~30 epochs each.

`cvmax_v2` uses an **absolute-RE-target gate** instead of the warmup-relative
gate that v1 uses.  The motivation: v1's `recon ≤ recon_warmup × budget`
threshold is self-tracking (`recon_warmup` is captured *during* each ISTA
call, after which `recon` typically continues to decrease), so the gate is
almost never closed in practice — v1 effectively reduces to plain `cv`.

In v2, `recon_budget` is reinterpreted as a **target RE**: CV is only pushed
when `current_RE ≤ recon_budget`.  Sweeping `recon_budget` over
(0.40, 0.50, 0.55, 0.60) forces the dictionary to first reach a low RE
through pure reconstruction; only then does CV pressure kick in.  This is
the configuration that lets cvmax actually compete with `null` at low RE.

Run only the v2 additions on top of existing Null/Cauchy records with
`bash p3_sweep.sh --methods cvmax_v2`.

The Pareto curves let us read off, at any fixed RE level, which method
produces the sparsest code and the highest reconstruction--activation
efficiency (RAE), and at any fixed mean activation which method gives the
lowest RE or highest RAE.  CV is the *mechanism* and reported diagnostically;
**RAE = -log2(RE) / mean |a| is a dimensionless engineering score** under the
common activation normalization.  It is not mutual information.  Historical
filenames and record keys retain `ICE` for backward compatibility.


## Files

```
p3_matched/
├── p3_train.py     # one (method, knob, seed) run end-to-end
├── p3_sweep.sh     # default current-v2 launcher (96 runs)
├── p3_eval.py      # aggregate JSON records → CSV tables + Pareto plot
└── README.md       # this file
results/
├── records/<run_id>.json     # per-run scalar metrics
├── checkpoints/<run_id>.pth  # trained model state_dict
├── log/<run_id>.log          # console log
└── eval/                     # produced by p3_eval.py
    ├── p3_long.csv
    ├── p3_summary.csv
    ├── p3_matched_RE.csv
    ├── p3_matched_a.csv
    ├── p3_pareto.pdf
    └── p3_ice_cv.pdf
```

`<run_id>` is `{method}__lmda{lmda}_a{alpha_s}_cv{cv_ratio}__seed{seed}`,
e.g. `cvmax__lmda0.005_a0_cv0.1__seed2`.


## Prerequisites

The training driver imports the existing Fig 6 model code:

```
Fig6/src/model/SparseNetNew.py
Fig6/src/model/ImageDataset1.py
Fig6/src/model/IMAGES.mat        # natural-image patches
```

Python deps: `torch`, `numpy`, `scipy`, `matplotlib`, plus whatever
`SparseNetNew.py` already requires.  The training script auto-detects CUDA
and falls back to CPU if no GPU is visible.


## Server workflow

### 1.  One-shot test (~1 min on GPU)

```
cd Fig6/p3_matched
python p3_train.py --method cvmax_v2 --lmda 5e-3 --cv_ratio 0.10 \
                   --seed 0 --epochs 2
```

This validates the environment, writes one JSON record, one checkpoint,
and one log.

### 2.  Full sweep

Sequential (recommended for a single GPU):

```
bash p3_sweep.sh
```

Parallel (e.g. 4 GPUs or a multi-process box; each child claims whatever
device PyTorch picks):

```
bash p3_sweep.sh --parallel 4
```

Useful flags:

```
--epochs 5            # quick run with fewer epochs
--seeds 0,1           # two seeds: 64 current-v2 runs
--methods cvmax_v2    # restrict to the current CVM method
--methods all         # also include the legacy v1 sensitivity
--dry-run             # print commands without executing
```

The sweep is **resumable**: if `results/records/<run_id>.json` already
exists for a given configuration, the sweep skips it.  Killing the
script and re-launching picks up where it left off.

### 3.  Aggregate and analyse

```
python p3_eval.py
```

Produces:

* **`results/eval/p3_long.csv`** — one row per (method, knob, seed) run
  with all scalar metrics.
* **`results/eval/p3_summary.csv`** — collapsed across seeds (mean ± std
  of every metric for every (method, knob)).
* **`results/eval/p3_matched_RE.csv`** — for each RE target, what mean |a|
  does each method need? (per-budget rows + an envelope `cvmax` row that
  picks the best budget at each target).
* **`results/eval/p3_matched_RE_ICE.csv`** — for each RE target, what RAE
  does each method achieve? (legacy filename; this is the **headline outcome metric**).
* **`results/eval/p3_matched_a.csv`** — for each mean |a| target, what RE
  does each method achieve?
* **`results/eval/p3_matched_a_ICE.csv`** — for each mean |a| target, what
  RAE does each method achieve? (legacy filename).
* **`results/eval/p3_pareto.pdf`** — RE vs. mean |a| Pareto; null and cauchy
  are single curves, cvmax is one curve per `recon_budget` level.
* **`results/eval/p3_ice_vs_re.pdf`** — RAE vs. RE, the matched-RE comparison
  axis (legacy filename; this is what we report in the manuscript).
* **`results/eval/p3_ice_cv.pdf`** — RAE vs. CV (left) and CV vs. knob
  (right; legacy filename), diagnostic plot showing the mechanism.

The targets default to the 25/50/75 percentiles of the observed metric
range, so the script is self-bootstrapping; override with
`--re_targets 0.05,0.07,0.10` etc. for the values you actually want
quoted in the manuscript.


## Per-run metric record

Every JSON record contains, in addition to the full config and per-epoch
training trace:

```python
"eval": {
    "RE":                  ...,   # ||x - Ua||_2 / ||x||_2 averaged over batches
    "mean_a":              ...,   # mean |a|
    "CV_pop":              ...,   # population CV (per patch, then averaged)
    "CV_lt":               ...,   # lifetime CV (per neuron, then averaged)
    "entropy_per_neuron":  ...,   # 200-bin H(|a|) averaged across K neurons
    "ICE_log_a":           ...,   # legacy key: RAE = -log2(RE) / mean_a
    "ICE_ent_a":           ...,   # legacy diagnostic: entropy_per_neuron / mean_a
}
```

The `eval` block is computed on a fresh deterministic pass after training
finishes, so it is comparable across runs even if mid-training noise
differed.


## Sweep grid (kept identical across seeds)

```
LMDAS_NULL    = (1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2)  # 8
ALPHAS_CAUCHY = (0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)        # 8
RECON_TARGETS_V2 = (0.40, 0.50, 0.55, 0.60)                       # 4
CVRS_V2          = (0.05, 0.10, 0.20, 0.40)                       # 4
SEEDS         = (0, 1, 2)                                         # 3
```

Common (matched) settings:

```
patch_size  = 16
n_neuron    = 144  (an undercomplete dictionary for 256-D patches)
batch_size  = 1024
epochs      = 30   (chosen via convergence diagnostic — see below)
lr          = 1e-2 (SGD on U)
r_lr        = 1e-2 (ISTA step size for a)
lmda_base   = 5e-3 (held fixed for Cauchy and CVM-v2 sweeps)
```

The 30-epoch default came from `p3_converge_check.py`: at 30 epochs, RE
drift from epoch 30 → 50 is < 1% for null/cauchy and ~0.4% for cvmax, so
30 captures the bulk of the learning curve without paying for the
diminishing returns past 30.  Override on the command line if needed.


## Estimated cost

* Single run, 30 epochs, 144 neurons, 16×16 patches, batch 1024:
  ~3–5 min on an A100 / 3090.
* Full sweep (96 runs, sequential, single GPU): ~5–8 h.
* With `--parallel 4` on a 4-GPU box: ~1.5–2 h.


## Caveat — old records from before route A

If old v1 records share the directory, the default `p3_eval.py` method filter
excludes them. Use a clean output directory for a fresh archival rerun; do not
mix legacy records into custom aggregates made with `--methods all`.

```
python p3_eval.py --methods null,cauchy,cvmax_v2
```

Old records had a different `run_id` format and do not collide with v2; the
explicit method filter prevents silent algorithm mixing.


## Reproducibility

* `set_seed()` covers `random`, `numpy`, `torch` (incl. CUDA).
* Each run is independent: no global state, no shared file writes
  besides separate `<run_id>` files.
* The sweep records the full `vars(args)` in every JSON, so any record
  can be re-launched verbatim by reading its `config` block.
