"""
Build sparsenet_compare_results.json from Fig6 TensorBoard event files.

Output schema (matches fig2's existing consumer):
{
  "configs": ["baseline", "cauchy", "cv"],
  "labels":  {...},
  "hists":   {<config>: {epoch, ice, ice_std, cv_act, cv_act_std, recon, recon_std}},
  "n_seeds": 2,
  "_meta":   {derivation notes}
}

Mapping (after introspecting TB tags: ['Loss', 'E', 'CV_pop', 'CV_life', 'S']):
  recon  := E      (sum-squared reconstruction loss per epoch)
  cv_act := CV_pop (population CV of |a| at the end of each epoch)
  ice    := derived; see ICE_FORMULA below

ICE formula
-----------
    RE  = sqrt(E / IMG_NORM_SQ)        # relative reconstruction error
    ice = -log2(RE) / mean_abs_a       # bits / mean-activation

IMG_NORM_SQ is the batch-summed image norm-squared ||I||^2, which is a
constant of the training setup (batch_size=1024, patch_size=16, natural
image patches normalised the same way for all three conditions).
Calibrated from the v2 main-experiment final-epoch RE/E pairs across
the three conditions:

    baseline:  RE=0.611, E=7222 → ||I||^2 = E / RE^2 = 19354
    cauchy:    RE=0.702, E=9679 → ||I||^2                = 19632
    cv (v2):   RE=0.625, E=7788 → ||I||^2                = 19929
    mean ≈ 19638 (IQR width ~575); we hard-code 19500.

This formula reproduces the main-experiment ICE values to <2% (e.g.
baseline 12.32 vs reported 12.26).

An earlier version of this script used  -log2(E / E[0]) / mean|a|, which
measures *training improvement in recon* rather than absolute coding
efficiency.  That penalised cv_constrained_v2 because its 20-iteration
ISTA warmup means epoch-0 E is already low (less head-room to
"improve"), even though final RE is comparable to baseline.

mean|a| is read directly from the per-run training record `<output_dir>/
record<suffix>_<n>.mat` field `mu`, which train_new.py writes once per
epoch (line 199 of train_new.py).

cv condition
------------
The cv condition uses the final cv_constrained_v2 algorithm
(absolute-RE-gated CV-max — see Methods §CV-maximization algorithm).
train_new.py emits TB events to ./runs_cv_v2_new/sparsenet_N when
trained with --cons cv_constrained_v2 (the v2 name_map entry).  The
plain `cv` runs in runs_cv_16_nl_new are not used: they diverge in
batch-pooled training because they lack the recon-budget gate, and
were superseded by v2.
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / 'sparsenet_compare_results.json'
RECORD_DIR = HERE / 'record'  # train_new.py default output_dir

CONFIGS = ['baseline', 'cauchy', 'cv']
LABELS = {
    'baseline': 'L1 (Olshausen)',
    'cauchy':   'Cauchy (NL)',
    'cv':       'CV (ours)',
}
RUN_DIRS = {
    'baseline': [HERE / 'runs_16_new'             / f'sparsenet_{s}' for s in (0, 1)],
    'cauchy':   [HERE / 'runs_cauchy_16_nl_new'   / f'sparsenet_{s}' for s in (0, 1)],
    # cv = our final algorithm (cv_constrained_v2): the absolute-RE-gated
    # variant trained via train_new.py --cons cv_constrained_v2.
    # train_new.py's name_map sends events to ./runs_cv_v2_new/sparsenet_N.
    # Plain `cv` (runs_cv_16_nl_new) diverges to NaN and was never the
    # comparison target — see Methods §CV-maximization algorithm.
    'cv':       [HERE / 'runs_cv_v2_new'          / f'sparsenet_{s}' for s in (0, 1)],
}

# .mat suffix per condition (train_new.py: record<suffix>_<n>.mat)
RECORD_SUFFIX = {
    'baseline': '_16_new',
    'cauchy':   '_cauchy_16_nl_new',
    'cv':       '_cv_v2_new',
}

TAG_RECON = 'E'       # sum-squared reconstruction error per epoch
TAG_CV    = 'CV_pop'  # population CV of |a|

# Batch-summed ||I||^2 calibrated from main-experiment v2 RE/E ratios.
# RE = sqrt(E / IMG_NORM_SQ).  See module docstring for derivation.
IMG_NORM_SQ = 19500.0

N_GRID = 10           # 10 epochs per run (step 0..9)


def latest_events_file(run_dir: Path) -> Path | None:
    """Return the events file with the largest mtime (skip empty stubs)."""
    files = sorted(glob.glob(str(run_dir / 'events.out*')), key=os.path.getmtime)
    return Path(files[-1]) if files else None


def load_seed_series(ev_path: Path, mat_path: Path) -> dict[str, np.ndarray] | None:
    """Return {'steps', TAG_RECON, TAG_CV, 'mean_a'}.
    TAG_RECON and TAG_CV come from TB events; 'mean_a' comes from the .mat
    'mu' field (per-epoch mean |a|).  Returns None if any source is missing
    or all-NaN.
    """
    ea = EventAccumulator(str(ev_path), size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags()['scalars']
    if not tags:
        return None
    out: dict[str, np.ndarray] = {}
    steps_ref: np.ndarray | None = None
    for t in (TAG_RECON, TAG_CV):
        if t not in tags:
            return None
        scalars = ea.Scalars(t)
        steps = np.array([s.step for s in scalars], dtype=int)
        vals  = np.array([s.value for s in scalars], dtype=float)
        if steps_ref is None:
            steps_ref = steps
        out[t] = vals
    out['steps'] = steps_ref

    if not mat_path.exists():
        return None
    mat = loadmat(str(mat_path))
    if 'mu' not in mat:
        return None
    mu = np.asarray(mat['mu'], dtype=float).flatten()
    # If train_new.py wrote a different epoch count than TB, truncate to TB
    # length (TB is one entry per epoch; mu is per-epoch via record_mu.append)
    n = min(len(mu), len(out['steps']))
    if n == 0:
        return None
    out['mean_a'] = mu[:n]
    out[TAG_RECON] = out[TAG_RECON][:n]
    out[TAG_CV]    = out[TAG_CV][:n]
    out['steps']   = out['steps'][:n]
    return out


def is_all_nan(arr: np.ndarray) -> bool:
    return arr.size > 0 and np.all(np.isnan(arr))


def derive_ice(E: np.ndarray, mean_a: np.ndarray) -> np.ndarray:
    """ice = -log2(RE) / mean|a|, with RE = sqrt(E / IMG_NORM_SQ).

    This matches the main-experiment definition (result_new.py: ice =
    -np.log2(re_avg) / cost_avg).  IMG_NORM_SQ is a batch-summed
    ||I||^2 calibrated from final-epoch RE/E pairs (see module
    docstring); it is constant across conditions because all three
    methods see the same training data and batch size.
    Returns NaN where derivation is undefined.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        re = np.sqrt(np.maximum(E, 0.0) / IMG_NORM_SQ)
        log_term = np.where(re > 0, -np.log2(re), np.nan)
        ice = np.where(mean_a > 0, log_term / mean_a, np.nan)
    return ice


def collect_condition(config: str) -> dict:
    """Aggregate seeds for one condition into the JSON shape."""
    series_per_seed = []
    diagnostics = []
    suffix = RECORD_SUFFIX[config]
    for seed_idx, rd in enumerate(RUN_DIRS[config]):
        ev = latest_events_file(rd)
        if ev is None:
            diagnostics.append(f'{rd.name}: no events file')
            series_per_seed.append(None)
            continue
        mat_path = RECORD_DIR / f'record{suffix}_{seed_idx}.mat'
        s = load_seed_series(ev, mat_path)
        if s is None:
            diagnostics.append(
                f'{rd.name}: missing data ({ev.name} or '
                f'record{suffix}_{seed_idx}.mat)'
            )
            series_per_seed.append(None)
            continue
        # check NaN in any required series
        if (is_all_nan(s[TAG_RECON]) or is_all_nan(s[TAG_CV])
                or is_all_nan(s['mean_a'])):
            diagnostics.append(
                f'{rd.name}: training diverged (all-NaN in {ev.name})'
            )
            series_per_seed.append(None)
            continue
        diagnostics.append(
            f'{rd.name}: OK ({ev.name}, {len(s["steps"])} epochs, mu from .mat)'
        )
        series_per_seed.append(s)

    # Common epoch grid: union of seed step-ranges → linspace
    all_steps = [s['steps'] for s in series_per_seed if s is not None]
    if not all_steps:
        # no seed survived → return all-NaN block
        grid = list(range(N_GRID))
        nan = [None] * N_GRID
        return {
            'epoch':      grid,
            'ice':        nan,    'ice_std':    nan,
            'cv_act':     nan,    'cv_act_std': nan,
            'recon':      nan,    'recon_std':  nan,
            'n_valid_seeds': 0,
            '_diagnostics': diagnostics,
        }

    smin = max(s.min() for s in all_steps)
    smax = min(s.max() for s in all_steps)
    grid = np.linspace(smin, smax, min(N_GRID, smax - smin + 1)).astype(int)

    metrics: dict[str, list[np.ndarray]] = {'ice': [], 'cv_act': [], 'recon': []}
    for s in series_per_seed:
        if s is None:
            continue
        E = s[TAG_RECON]; CV = s[TAG_CV]; mean_a = s['mean_a']
        ice = derive_ice(E, mean_a)
        for key, src in (('ice', ice), ('cv_act', CV), ('recon', E)):
            interp = np.interp(grid, s['steps'], src,
                               left=np.nan, right=np.nan)
            metrics[key].append(interp)

    out = {'epoch': grid.tolist(), 'n_valid_seeds': len(metrics['ice']),
           '_diagnostics': diagnostics}
    for key, stack in metrics.items():
        arr = np.vstack(stack)  # (n_valid, n_grid)
        mean = np.nanmean(arr, axis=0)
        std = (np.nanstd(arr, axis=0, ddof=1)
               if arr.shape[0] > 1 else np.zeros_like(mean))
        # Convert NaN → None for JSON compatibility
        out[key]          = [None if math.isnan(v) else float(v) for v in mean]
        out[f'{key}_std'] = [None if math.isnan(v) else float(v) for v in std]
    return out


def main() -> int:
    hists = {cfg: collect_condition(cfg) for cfg in CONFIGS}
    out = {
        'configs':  CONFIGS,
        'labels':   LABELS,
        'hists':    hists,
        'n_seeds':  2,
        '_meta': {
            'recon_tag': TAG_RECON,
            'cv_act_tag': TAG_CV,
            'mean_a_source': 'record<suffix>_<seed>.mat field "mu"',
            'ice_formula': '-log2(sqrt(E / IMG_NORM_SQ)) / mean|a|',
            'img_norm_sq': IMG_NORM_SQ,
            'ice_note':
                ('Matches the main-experiment ICE definition '
                 '(result_new.py: -log2(RE) / mean|a|, with RE = '
                 '||I - I_hat||_2 / ||I||_2).  IMG_NORM_SQ is the batch-summed '
                 '||I||^2, calibrated from main-experiment v2 RE/E ratios '
                 '(baseline, cauchy, cv_v2 all give 19354/19632/19929 → '
                 '19500 hard-coded).  Reproduces the main-experiment final-'
                 'epoch ICE values to <2%.  Earlier versions of this script '
                 'used -log2(E/E[0]) / mean|a|, which conflated training '
                 'progress with coding efficiency and penalised cvmax_v2 '
                 'because its ISTA warmup makes epoch-0 E artificially low.'),
            'epoch_axis_note':
                'TB step axis used directly as epoch index (each scalar logged '
                'once per epoch by the SparseNet training loop).',
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f'wrote {OUT_PATH}')

    # ── Verification ──────────────────────────────────────────────────────
    print('\n=== verification ===')
    print(f'configs: {out["configs"]}')
    for cfg in CONFIGS:
        h = hists[cfg]
        print(f'\n[{cfg}] n_valid_seeds={h["n_valid_seeds"]}')
        for diag in h['_diagnostics']:
            print(f'  {diag}')
        last = lambda k: h[k][-1] if h[k][-1] is not None else 'NaN'
        print(f'  final epoch values: ice={last("ice")}  '
              f'cv_act={last("cv_act")}  recon={last("recon")}')

    # Punchline checks (only for valid data)
    print('\n=== punchline checks ===')
    final_cv = {c: hists[c]['cv_act'][-1] for c in CONFIGS}
    final_ice = {c: hists[c]['ice'][-1] for c in CONFIGS}
    print(f'final CV_act:  {final_cv}')
    print(f'final ICE:     {final_ice}')
    if final_cv['cv'] is not None:
        if final_cv['cv'] > final_cv['baseline'] and final_cv['cv'] > final_cv['cauchy']:
            print('  ✓ cv > baseline AND cv > cauchy on cv_act')
        else:
            print('  ✗ cv NOT highest on cv_act (may indicate weak regulariser)')
        if final_ice['cv'] > final_ice['baseline']:
            print('  ✓ cv > baseline on ICE')
        else:
            print('  ✗ cv NOT > baseline on ICE')
    else:
        print('  cv condition is NaN — punchline checks skipped')
        print('  → re-train cv with cv_constrained_v2 mode, or drop cv from Fig 2')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
