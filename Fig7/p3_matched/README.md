# P3 — Matched-baseline analysis for Fig. 7

This subdirectory implements the **matched-RE / matched-activation baseline**
that the reviewer asked for as the P3 priority item.  It compares three
sparse-coding training regimes on the natural-image patch dataset, holding
every architectural and training hyper-parameter equal except for the
regularisation knob:

| method     | objective              | SparseNet `cons` mode      | swept knob             |
|------------|------------------------|----------------------------|------------------------|
| `null`     | reconstruction + L1    | `non`                      | L1 strength `lmda` (8 levels) |
| `cauchy`   | reconstruction + L1 + Cauchy prior | `s`            | Cauchy weight `alpha_s` (8 levels) |
| `cvmax`    | reconstruction + L1 + budget-gated CV penalty | **`cv_constrained`**     | 2-D grid: `recon_budget` (4) × `cv_ratio` (4) |
| `cvmax_v2` | reconstruction + L1 + **absolute-RE-target gated CV penalty** | **`cv_constrained_v2`** | 2-D grid: `recon_budget` (4 target REs: 0.4, 0.5, 0.55, 0.6) × `cv_ratio` (4) |

`cvmax` uses **`cv_constrained`** — the same mode the main experiment uses
(see [`Fig7/result_new.py`](../result_new.py): `USE_CONSTRAINED=True`).
The plain `cv` mode was an early prototype that cannot match `null`'s RE
because it has no reconstruction-budget gate; using it for matched-baseline
caused `cvmax` to plateau at RE ≈ 0.71 even at the smallest `cv_ratio`,
making the comparison unfair.  `cv_constrained` gates the CV penalty so
it only fires when reconstruction is within `recon_budget × recon_warmup`,
letting `cvmax` track `null`'s RE while still pushing CV up.

Sweep totals:
- null:     8 × 3 seeds        = 24 runs
- cauchy:   8 × 3 seeds        = 24 runs
- cvmax:    4 × 4 × 3 seeds    = 48 runs
- cvmax_v2: 4 × 4 × 3 seeds    = 48 runs
- **Total: 144 runs**, ~30 epochs each.

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

Run only the v2 additions on top of an existing 96-run sweep with
`bash p3_sweep.sh --methods cvmax_v2`.

The Pareto curves let us read off, at any fixed RE level, which method
produces the sparsest code (matched-RE) and the highest ICE (matched-RE-ICE),
and at any fixed mean activation which method gives the lowest RE (matched-a)
or highest ICE (matched-a-ICE).  CV is the *mechanism* and reported
diagnostically; **ICE = bits/activation is the outcome metric** for the
matched comparison.


## Files

```
p3_matched/
├── p3_train.py     # one (method, knob, seed) run end-to-end
├── p3_sweep.sh     # bash launcher that fires all 72 runs
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

The training driver imports the existing Fig 7 model code:

```
Fig7/src/model/SparseNetNew.py
Fig7/src/model/ImageDataset1.py
Fig7/src/model/IMAGES.mat        # natural-image patches
```

Python deps: `torch`, `numpy`, `scipy`, `matplotlib`, plus whatever
`SparseNetNew.py` already requires.  The training script auto-detects CUDA
and falls back to CPU if no GPU is visible.


## Server workflow

### 1.  One-shot test (~1 min on GPU)

```
cd Fig7/p3_matched
python p3_train.py --method cvmax --lmda 5e-3 --cv_ratio 0.10 \
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
--seeds 0,1           # only 2 seeds × 3 methods × 8 knobs = 48 runs
--methods cvmax       # restrict to one method
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
* **`results/eval/p3_matched_RE_ICE.csv`** — for each RE target, what ICE
  does each method achieve? (this is the **headline outcome metric**).
* **`results/eval/p3_matched_a.csv`** — for each mean |a| target, what RE
  does each method achieve?
* **`results/eval/p3_matched_a_ICE.csv`** — for each mean |a| target, what
  ICE does each method achieve?
* **`results/eval/p3_pareto.pdf`** — RE vs. mean |a| Pareto; null and cauchy
  are single curves, cvmax is one curve per `recon_budget` level.
* **`results/eval/p3_ice_vs_re.pdf`** — ICE vs. RE, the matched-RE comparison
  axis (this is what we report in the manuscript).
* **`results/eval/p3_ice_cv.pdf`** — ICE vs. CV (left) and CV vs. knob
  (right), diagnostic plot showing the mechanism.

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
    "ICE_log_a":           ...,   # -log2(RE) / mean_a   bits / activation
    "ICE_ent_a":           ...,   # entropy_per_neuron / mean_a
}
```

The `eval` block is computed on a fresh deterministic pass after training
finishes, so it is comparable across runs even if mid-training noise
differed.


## Sweep grid (kept identical across seeds)

```
LMDAS_NULL    = (1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2)  # 8
ALPHAS_CAUCHY = (0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)        # 8
RECON_BUDGETS = (1.0, 1.05, 1.10, 1.20)                           # 4
CVRS_CVMAX    = (0.05, 0.10, 0.20, 0.40)                          # 4
SEEDS         = (0, 1, 2)                                         # 3
```

Common (matched) settings:

```
patch_size  = 16
n_neuron    = 144  (i.e. 9× overcomplete on 16×16 patches)
batch_size  = 1024
epochs      = 30   (chosen via convergence diagnostic — see below)
lr          = 1e-2 (SGD on U)
r_lr        = 1e-2 (ISTA step size for a)
lmda_base   = 5e-3 (held fixed for cauchy and cvmax sweeps)
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

If you have existing JSON records from a sweep that used the plain `cv`
mode (run before recon_budget was added to `p3_train.py`), delete them
before re-running:

```
rm -rf results/records results/checkpoints results/eval
```

Old records had a different `run_id` format (no `_b` budget suffix) so
they will not collide with the new sweep, but `p3_eval.py` would mix
them in and contaminate the comparison.  Cleanest path is a full reset.


## Reproducibility

* `set_seed()` covers `random`, `numpy`, `torch` (incl. CUDA).
* Each run is independent: no global state, no shared file writes
  besides separate `<run_id>` files.
* The sweep records the full `vars(args)` in every JSON, so any record
  can be re-launched verbatim by reading its `config` block.
