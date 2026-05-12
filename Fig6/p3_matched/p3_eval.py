"""
P3 matched-baseline aggregator and Pareto / matched-RE / matched-a analysis.

Loads every JSON record produced by p3_train.py and produces:

  1. A long-form CSV of every (method, knob, seed) row (one per run).
  2. A summary CSV: per-(method, knob) means + std across seeds.
  3. A Pareto plot RE vs. mean_a (one curve per method) -- the basis of
     all matched-baseline statements.
  4. Matched-RE table: at a fixed RE target, what is each method's
     mean_a (smaller = sparser at the same reconstruction)?
  5. Matched-a  table: at a fixed mean_a target, what is each method's
     RE (smaller = better reconstruction at the same activation)?
  6. A second plot ICE vs. CV showing the metabolic-efficiency / CV
     relationship that motivates the CV-max method.

Usage
-----
  python p3_eval.py
  python p3_eval.py --records_dir results/records --out_dir results/eval
  python p3_eval.py --re_targets 0.05,0.07,0.10 --a_targets 0.02,0.05,0.10

The matched targets default to a few percentiles of the observed RE / mean_a
distribution if not supplied, so the script is self-bootstrapping.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # non-GUI backend — saves figures to disk, no Qt/GTK needed
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
RECORDS_DIR = RESULTS / 'records'
EVAL_DIR = RESULTS / 'eval'

METHOD_ORDER = ['null', 'cauchy', 'cvmax', 'cvmax_v2']
METHOD_LABEL = {
    'null':     'Null (L1 only)',
    'cauchy':   'Cauchy prior',
    'cvmax':    'CV (cv_constrained)',
    'cvmax_v2': 'CV-v2 (absolute-RE gate)',
}
METHOD_COLOR = {'null': '0.4', 'cauchy': 'C0',
                'cvmax': 'C3', 'cvmax_v2': 'C2'}

# cvmax v1 sweeps recon_budget along curves (each budget a curve, cv_ratio
# the within-curve knob — RE walks).  v2 is the opposite: recon_budget pins
# RE, so each cv_ratio value is a curve and budget is the within-curve knob.
SUBMETHOD_COLORMAP = {
    'cvmax': {  # keyed by recon_budget
        1.0:  '#b30000',
        1.05: '#e34a33',
        1.10: '#fc8d59',
        1.20: '#fdcc8a',
    },
    'cvmax_v2': {  # keyed by cv_ratio
        0.05: '#a1d99b',
        0.10: '#41ab5d',
        0.20: '#238b45',
        0.40: '#005a32',
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='P3 matched-baseline aggregator')
    p.add_argument('--records_dir', type=str, default=str(RECORDS_DIR))
    p.add_argument('--out_dir',     type=str, default=str(EVAL_DIR))
    p.add_argument('--re_targets',  type=str, default='',
                   help='Comma-separated RE values for matched-RE table.  '
                        'Empty -> use 25/50/75 pct of observed RE range.')
    p.add_argument('--a_targets',   type=str, default='',
                   help='Comma-separated mean_a values for matched-a table.  '
                        'Empty -> 25/50/75 pct of observed mean_a range.')
    p.add_argument('--ice', type=str, default='ICE_log_a',
                   choices=['ICE_log_a', 'ICE_ent_a'],
                   help='Which ICE definition to use for the ICE vs. CV plot.')
    return p.parse_args()


def knob_for(rec: dict) -> float:
    """Return the active sweep knob for a given record."""
    m = rec['method']
    cfg = rec['config']
    if m == 'null':                  return float(cfg['lmda'])
    if m == 'cauchy':                return float(cfg['alpha_s'])
    if m in ('cvmax', 'cvmax_v2'):   return float(cfg['cv_ratio'])
    raise ValueError(f'unknown method {m}')


def knob_label(method: str) -> str:
    return {'null': 'lmda', 'cauchy': 'alpha_s',
            'cvmax': 'cv_ratio', 'cvmax_v2': 'cv_ratio'}[method]


def load_records(records_dir: Path) -> list[dict]:
    out = []
    for jf in sorted(records_dir.glob('*.json')):
        with open(jf) as f:
            rec = json.load(f)
        out.append(rec)
    return out


def to_long_rows(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        ev = rec['eval']
        cfg = rec['config']
        rb = float(cfg.get('recon_budget', 1.0))
        rows.append({
            'run_id': rec['run_id'],
            'method': rec['method'],
            'knob_name': knob_label(rec['method']),
            'knob': knob_for(rec),
            'recon_budget': rb,
            'seed': cfg['seed'],
            'lmda': cfg['lmda'],
            'alpha_s': cfg['alpha_s'],
            'cv_ratio': cfg['cv_ratio'],
            'RE': ev['RE'],
            'mean_a': ev['mean_a'],
            'CV_pop': ev['CV_pop'],
            'CV_lt': ev['CV_lt'],
            'entropy_per_neuron': ev['entropy_per_neuron'],
            'ICE_log_a': ev['ICE_log_a'],
            'ICE_ent_a': ev['ICE_ent_a'],
            'train_time_sec': rec.get('train_time_sec', float('nan')),
        })
    return rows


def submethod_for(row: dict) -> str:
    """Pareto curves are drawn per-submethod.

    cvmax  (v1): RE is sweep-controlled by cv_ratio (per budget), so each
                 recon_budget gets its own RE-vs-mean_a curve traced by
                 cv_ratio.  Group by recon_budget.
    cvmax_v2:    RE is *pinned* by recon_budget (the absolute-RE gate), so
                 cv_ratio barely moves RE; the curve sweep direction is
                 actually budget.  Group by cv_ratio so each curve traces
                 RE across budgets.
    null/cauchy: single curve each.
    """
    if row['method'] == 'cvmax':
        return f"cvmax_b{row['recon_budget']:g}"
    if row['method'] == 'cvmax_v2':
        return f"cvmax_v2_cv{row['knob']:g}"
    return row['method']


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate_seeds(rows: list[dict]) -> list[dict]:
    """Per-(method, recon_budget, knob) summary across seeds.

    For null/cauchy recon_budget is always 1.0, so the keying still
    collapses to (method, knob) for them.  For cvmax this is a 2-D
    grid: (cvmax, budget, cv_ratio).
    """
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r['method'], r['recon_budget'], r['knob'])
        by_key.setdefault(key, []).append(r)
    summary = []
    metric_keys = ['RE', 'mean_a', 'CV_pop', 'CV_lt',
                   'entropy_per_neuron', 'ICE_log_a', 'ICE_ent_a']
    for (method, rb, knob), grp in sorted(by_key.items()):
        row = {'method': method,
               'submethod': submethod_for(grp[0]),
               'knob_name': knob_label(method),
               'knob': knob,
               'recon_budget': rb,
               'n_seeds': len(grp)}
        for k in metric_keys:
            vals = np.array([g[k] for g in grp], dtype=float)
            row[f'{k}_mean'] = float(np.mean(vals))
            row[f'{k}_std']  = float(np.std(vals, ddof=1) if len(vals) > 1
                                     else 0.0)
        summary.append(row)
    return summary


def sort_by(records: list[dict], key: str) -> list[dict]:
    return sorted(records, key=lambda r: r[key])


def interp_at(curve_x: np.ndarray, curve_y: np.ndarray,
              targets: np.ndarray) -> np.ndarray:
    """Sort by x, then linear-interpolate y at each target.

    Out-of-range targets return NaN (no extrapolation).
    """
    order = np.argsort(curve_x)
    cx, cy = curve_x[order], curve_y[order]
    out = np.full_like(targets, np.nan, dtype=float)
    for i, t in enumerate(targets):
        if t < cx.min() or t > cx.max():
            continue
        out[i] = float(np.interp(t, cx, cy))
    return out


def matched_table(summary: list[dict], x_metric: str, y_metric: str,
                  targets: np.ndarray, label: str,
                  better: str = 'min') -> list[dict]:
    """For each method, interpolate y_metric_mean at fixed x_metric_mean=target.

    For cvmax, sweep is 2-D (recon_budget × cv_ratio): we report both
    (a) the per-budget curve interpolated value, and
    (b) the envelope (best across budgets) labelled as method='cvmax'.

    The envelope direction is set by `better`:
      - better='min' (default): smaller y is better (e.g. mean_a at fixed RE,
        RE at fixed mean_a)
      - better='max':           larger  y is better (e.g. ICE at fixed RE)
    """
    rows = []
    for method in METHOD_ORDER:
        per_method = [r for r in summary if r['method'] == method]
        if not per_method:
            continue

        if method in ('cvmax', 'cvmax_v2'):
            # cvmax v1: each recon_budget gives an RE-vs-mean_a curve traced
            #           by cv_ratio.  Group by budget.
            # cvmax v2: recon_budget pins RE; cv_ratio barely moves RE.  The
            #           sweep direction is budget, so group by cv_ratio so
            #           each curve traces RE across budgets.
            if method == 'cvmax':
                group_key = 'recon_budget'
                group_tag = 'b'
                group_values = sorted({r['recon_budget'] for r in per_method})
            else:  # cvmax_v2
                group_key = 'knob'  # cv_ratio
                group_tag = 'cv'
                group_values = sorted({r['knob'] for r in per_method})

            per_group_y = []
            for gv in group_values:
                sub = [r for r in per_method if r[group_key] == gv]
                cx = np.array([r[f'{x_metric}_mean'] for r in sub])
                cy = np.array([r[f'{y_metric}_mean'] for r in sub])
                cy_std = np.array([r[f'{y_metric}_std'] for r in sub])
                ys = interp_at(cx, cy, targets)
                ys_std = interp_at(cx, cy_std, targets)
                per_group_y.append((gv, ys, ys_std))
                for t, yv, ye in zip(targets, ys, ys_std):
                    rows.append({
                        'comparison': label,
                        'method': f'{method}_{group_tag}{gv:g}',
                        'recon_budget':
                            gv if group_key == 'recon_budget' else None,
                        f'{x_metric}_target': float(t),
                        f'{y_metric}_at_target':
                            float(yv) if np.isfinite(yv) else None,
                        f'{y_metric}_std_at_target':
                            float(ye) if np.isfinite(ye) else None,
                    })
            # envelope across group_values — guard against targets that fall
            # outside every group's interpolation range (entire column NaN).
            yarr = np.vstack([y for _, y, _ in per_group_y])  # (G, T)
            sarr = np.vstack([s for _, _, s in per_group_y])
            T = yarr.shape[1]
            envelope = np.full(T, np.nan)
            best_idx = np.full(T, -1, dtype=int)
            for i in range(T):
                col = yarr[:, i]
                if np.all(np.isnan(col)):
                    continue
                envelope[i] = (np.nanmin(col) if better == 'min'
                               else np.nanmax(col))
                best_idx[i] = int(np.nanargmin(col) if better == 'min'
                                  else np.nanargmax(col))
            envelope_std = np.array(
                [sarr[best_idx[i], i] if best_idx[i] >= 0 else np.nan
                 for i in range(T)])
            envelope_group = np.array(
                [group_values[best_idx[i]] if best_idx[i] >= 0 else np.nan
                 for i in range(T)])
            for i, t in enumerate(targets):
                # For v1 envelope row, recon_budget = winning budget.
                # For v2 envelope row, recon_budget is None — the winning
                # cv_ratio is identifiable via the per-curve rows above.
                rows.append({
                    'comparison': label,
                    'method': method,  # envelope row
                    'recon_budget':
                        (float(envelope_group[i])
                         if group_key == 'recon_budget'
                         and np.isfinite(envelope_group[i]) else None),
                    f'{x_metric}_target': float(t),
                    f'{y_metric}_at_target':
                        float(envelope[i]) if np.isfinite(envelope[i]) else None,
                    f'{y_metric}_std_at_target':
                        float(envelope_std[i]) if np.isfinite(envelope_std[i])
                        else None,
                })
        else:
            cx = np.array([r[f'{x_metric}_mean'] for r in per_method])
            cy = np.array([r[f'{y_metric}_mean'] for r in per_method])
            cy_std = np.array([r[f'{y_metric}_std'] for r in per_method])
            ys = interp_at(cx, cy, targets)
            ys_std = interp_at(cx, cy_std, targets)
            for t, yv, ye in zip(targets, ys, ys_std):
                rows.append({
                    'comparison': label,
                    'method': method,
                    'recon_budget': None,
                    f'{x_metric}_target': float(t),
                    f'{y_metric}_at_target': float(yv) if np.isfinite(yv) else None,
                    f'{y_metric}_std_at_target':
                        float(ye) if np.isfinite(ye) else None,
                })
    return rows


def percentiles_target(summary: list[dict], metric: str,
                       qs=(25, 50, 75)) -> np.ndarray:
    vals = np.array([r[f'{metric}_mean'] for r in summary], dtype=float)
    return np.percentile(vals, qs)


def parse_targets(arg: str, fallback: np.ndarray) -> np.ndarray:
    arg = arg.strip()
    if not arg:
        return fallback
    return np.array([float(x) for x in arg.split(',') if x], dtype=float)


# -- Plots --------------------------------------------------------------------
def _submethod_groups(summary: list[dict]) -> dict[str, list[dict]]:
    """Split summary into one list per Pareto curve.

    null & cauchy each become one group; cvmax becomes one group per
    recon_budget level.
    """
    groups: dict[str, list[dict]] = {}
    for r in summary:
        groups.setdefault(r['submethod'], []).append(r)
    return groups


def _parse_submethod_param(submethod: str) -> float | None:
    """Extract the numeric param embedded in submethod name.
       'cvmax_b1.05'        -> 1.05  (budget for v1)
       'cvmax_v2_cv0.4'     -> 0.4   (cv_ratio for v2)
    """
    for sep in ('_b', '_cv'):
        if sep in submethod:
            try:
                return float(submethod.rsplit(sep, 1)[1])
            except ValueError:
                pass
    return None


def _color_for(submethod: str) -> str:
    if submethod == 'null':   return METHOD_COLOR['null']
    if submethod == 'cauchy': return METHOD_COLOR['cauchy']
    param = _parse_submethod_param(submethod)
    if submethod.startswith('cvmax_v2'):
        return SUBMETHOD_COLORMAP['cvmax_v2'].get(param,
                                                  METHOD_COLOR['cvmax_v2'])
    return SUBMETHOD_COLORMAP['cvmax'].get(param, METHOD_COLOR['cvmax'])


def _label_for(submethod: str) -> str:
    if submethod == 'null':   return METHOD_LABEL['null']
    if submethod == 'cauchy': return METHOD_LABEL['cauchy']
    param = _parse_submethod_param(submethod)
    if submethod.startswith('cvmax_v2'):
        return f'CV-v2  cv_ratio={param:g}'
    return f'CV  budget={param:g}'


def plot_pareto(summary: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for sm, grp in _submethod_groups(summary).items():
        grp = sorted(grp, key=lambda r: r['mean_a_mean'])
        x = np.array([r['mean_a_mean'] for r in grp])
        y = np.array([r['RE_mean']     for r in grp])
        yerr = np.array([r['RE_std']   for r in grp])
        xerr = np.array([r['mean_a_std'] for r in grp])
        ax.errorbar(x, y, yerr=yerr, xerr=xerr,
                    fmt='o-', ms=5, lw=1.5, capsize=2,
                    color=_color_for(sm),
                    label=_label_for(sm))
    ax.set_xlabel(r'mean $|a|$ (sparsity, lower = sparser)')
    ax.set_ylabel('Reconstruction error  RE')
    ax.set_title('P3 matched-baseline Pareto front')
    ax.legend(frameon=False, loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_ice_cv(summary: list[dict], ice_key: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for sm, grp in _submethod_groups(summary).items():
        grp = sorted(grp, key=lambda r: r['knob'])
        cv = np.array([r['CV_pop_mean']    for r in grp])
        ic = np.array([r[f'{ice_key}_mean'] for r in grp])
        knobs = np.array([r['knob']        for r in grp])
        axes[0].plot(cv, ic, 'o-', ms=5, lw=1.5,
                     color=_color_for(sm), label=_label_for(sm))
        axes[1].plot(knobs, cv, 'o-', ms=5, lw=1.5,
                     color=_color_for(sm), label=_label_for(sm))
    axes[0].set_xlabel('Population CV')
    axes[0].set_ylabel(f'ICE  ({ice_key})')
    axes[0].set_title('ICE vs. CV')
    axes[0].grid(alpha=0.3); axes[0].legend(frameon=False, fontsize=7,
                                           loc='best', ncol=2)

    axes[1].set_xlabel('Sweep knob (per-method)')
    axes[1].set_ylabel('Population CV')
    axes[1].set_title('CV vs. regularisation strength')
    axes[1].set_xscale('log')
    axes[1].grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_ice_vs_re(summary: list[dict], ice_key: str, out_path: Path,
                   re_min: float = 0.60) -> None:
    """ICE vs RE — the headline plot for matched-RE since ICE is the
    outcome metric and RE is the matching axis.

    The x-axis is cropped at re_min (default 0.60) to focus on the region
    where all three methods have data. Null's L1 sweep extends to lower
    RE (~0.37) at the cost of high mean activation; that under-regularised
    tail is omitted here and discussed in the caption.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for sm, grp in _submethod_groups(summary).items():
        grp = sorted(grp, key=lambda r: r['RE_mean'])
        x_full = np.array([r['RE_mean']         for r in grp])
        y_full = np.array([r[f'{ice_key}_mean'] for r in grp])
        e_full = np.array([r[f'{ice_key}_std']  for r in grp])
        mask = x_full >= re_min
        x, y, yerr = x_full[mask], y_full[mask], e_full[mask]
        if len(x) == 0:
            continue
        ax.errorbar(x, y, yerr=yerr, fmt='o-', ms=5, lw=1.5, capsize=2,
                    color=_color_for(sm), label=_label_for(sm))
    ax.axvline(0.60, ls='--', color='0.4', lw=0.9,
               label='matched-RE = 0.60')
    ax.set_xlim(left=re_min)
    ax.set_xlabel('Reconstruction error  RE')
    ax.set_ylabel(f'ICE  ({ice_key})  [bits/activation]')
    ax.set_title('ICE vs. RE  (matched-RE comparison axis)')
    ax.legend(frameon=False, loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# -- Main ---------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    records_dir = Path(args.records_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(records_dir)
    if not records:
        print(f'[p3_eval] no JSON records found in {records_dir}')
        return 1
    print(f'[p3_eval] loaded {len(records)} records from {records_dir}')

    rows = to_long_rows(records)
    write_csv(out_dir / 'p3_long.csv', rows)
    print(f'[p3_eval] wrote {out_dir/"p3_long.csv"}  ({len(rows)} rows)')

    summary = aggregate_seeds(rows)
    write_csv(out_dir / 'p3_summary.csv', summary)
    print(f'[p3_eval] wrote {out_dir/"p3_summary.csv"}  ({len(summary)} rows)')

    # Matched-RE: at fixed RE, compare mean_a (sparser = lower)
    re_targets = parse_targets(args.re_targets,
                               percentiles_target(summary, 'RE'))
    re_table = matched_table(summary, 'RE', 'mean_a',
                             re_targets, label='matched-RE',
                             better='min')
    write_csv(out_dir / 'p3_matched_RE.csv', re_table)
    print(f'[p3_eval] matched-RE targets: {re_targets.tolist()}')

    # Matched-RE on ICE: at fixed RE, compare ICE (higher = better)
    ice_re_table = matched_table(summary, 'RE', args.ice,
                                 re_targets, label='matched-RE-ICE',
                                 better='max')
    write_csv(out_dir / 'p3_matched_RE_ICE.csv', ice_re_table)

    # Matched-a: at fixed mean_a, compare RE (lower = better)
    a_targets = parse_targets(args.a_targets,
                              percentiles_target(summary, 'mean_a'))
    a_table = matched_table(summary, 'mean_a', 'RE',
                            a_targets, label='matched-a', better='min')
    write_csv(out_dir / 'p3_matched_a.csv', a_table)
    print(f'[p3_eval] matched-a  targets: {a_targets.tolist()}')

    # Matched-a on ICE: at fixed mean_a, compare ICE (higher = better)
    ice_a_table = matched_table(summary, 'mean_a', args.ice,
                                a_targets, label='matched-a-ICE',
                                better='max')
    write_csv(out_dir / 'p3_matched_a_ICE.csv', ice_a_table)

    # Plots
    plot_pareto(summary, out_dir / 'p3_pareto.pdf')
    print(f'[p3_eval] wrote {out_dir/"p3_pareto.pdf"}')
    plot_ice_cv(summary, args.ice, out_dir / 'p3_ice_cv.pdf')
    print(f'[p3_eval] wrote {out_dir/"p3_ice_cv.pdf"}')
    plot_ice_vs_re(summary, args.ice, out_dir / 'p3_ice_vs_re.pdf')
    print(f'[p3_eval] wrote {out_dir/"p3_ice_vs_re.pdf"}')

    # Console-friendly mini-tables (envelope rows only for cvmax / cvmax_v2)
    def envelope_only(rows: list[dict]) -> list[dict]:
        return [r for r in rows
                if r['method'] in ('null', 'cauchy', 'cvmax', 'cvmax_v2')]

    print('\n=== Matched-RE: mean_a at fixed RE (smaller = sparser) ===')
    _print_pivot(envelope_only(re_table),  'RE_target', 'mean_a_at_target')

    print('\n=== Matched-RE: ICE at fixed RE (larger = more efficient) ===')
    _print_pivot(envelope_only(ice_re_table), 'RE_target',
                 f'{args.ice}_at_target')

    print('\n=== Matched-a: RE at fixed mean_a (smaller = better recon) ===')
    _print_pivot(envelope_only(a_table), 'mean_a_target', 'RE_at_target')

    print('\n=== Matched-a: ICE at fixed mean_a (larger = more efficient) ===')
    _print_pivot(envelope_only(ice_a_table), 'mean_a_target',
                 f'{args.ice}_at_target')
    return 0


def _print_pivot(table: list[dict], x_col: str, y_col: str) -> None:
    if not table:
        print('  (empty)')
        return
    targets = sorted({row[x_col] for row in table})
    methods = [m for m in METHOD_ORDER
               if any(r['method'] == m for r in table)]
    header = f"{x_col:>12} | " + " ".join(f'{m:>10}' for m in methods)
    print(header)
    print('-' * len(header))
    for t in targets:
        cells = []
        for m in methods:
            v = next((r[y_col] for r in table
                      if r['method'] == m and r[x_col] == t), None)
            cells.append(f'{v:>10.4g}' if v is not None else f'{"--":>10}')
        print(f"{t:>12.4g} | " + " ".join(cells))


if __name__ == '__main__':
    raise SystemExit(main())
