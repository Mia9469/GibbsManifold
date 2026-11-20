#!/usr/bin/env python3
"""
Shared statistics helpers for the Fig6 generalization tests.

Design principle: the replication unit is defined by the test design, not by
per-batch checkpoints. Full-data structured tests use the training run as the
unit; held-out natural-image tests aggregate seeds within each image fold so
that the held-out image is the unit. Checkpoint averaging happens upstream, so
every value handed to these functions is one number per reported unit.

Report absolute mean differences and bootstrap intervals. Pooled-SD Cohen's d
is descriptive and does not establish practical importance. Image comparisons
preserve fold pairing; inference is conditional on this ten-image dataset,
whose LOO training sets overlap. Seed variability is not population-of-images
uncertainty. Multiplicity correction is within metric, over three model pairs.
"""

import numpy as np
from scipy.stats import ttest_ind, ttest_1samp


def cohens_d(a, b):
    """Standardized mean difference (b - a) with pooled SD (Hedges-free)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled < 1e-12:
        return np.inf * np.sign(b.mean() - a.mean()) if b.mean() != a.mean() else 0.0
    return (b.mean() - a.mean()) / pooled


def d_magnitude(d):
    """Cohen's qualitative label for |d|."""
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    if ad < 1.2:  return "large"
    return "very large"


def bootstrap_ci_mean(x, n_boot=10000, ci=95, seed=0):
    """Bootstrap CI for the mean of x."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return (lo, hi)


def bootstrap_ci_diff(a, b, n_boot=10000, ci=95, seed=0):
    """Bootstrap CI for the difference of means (b - a)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ba = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    bb = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    diff = bb - ba
    lo = np.percentile(diff, (100 - ci) / 2)
    hi = np.percentile(diff, 100 - (100 - ci) / 2)
    return (lo, hi)


def paired_values(rows, metric, first, second, unit):
    """Align complete paired observations by their scientific unit, never row order."""
    groups = {}
    for method in (first, second):
        selected = [r for r in rows if r['type'] == method]
        ids = [r[unit] for r in selected]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate {unit} for {method}')
        groups[method] = {r[unit]: float(r[metric]) for r in selected}
    if set(groups[first]) != set(groups[second]):
        raise ValueError(f'Unmatched {unit} identifiers: {first}, {second}')
    ids = sorted(groups[first])
    a = np.array([groups[first][i] for i in ids])
    b = np.array([groups[second][i] for i in ids])
    if len(ids) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Paired inference needs at least two finite complete pairs')
    return a, b


def pairwise_statistics(rows, metric, first, second, paired_unit=None,
                        family_size=3):
    if paired_unit is not None:
        a, b = paired_values(rows, metric, first, second, paired_unit)
        diff = b - a
        lo, hi = bootstrap_ci_mean(diff)
        if np.all(diff == 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = ttest_1samp(diff, 0)
        test = 'two-sided paired t-test on unit differences'
        bootstrap = 'percentile bootstrap of paired unit differences, seed 0, 10000 draws'
    else:
        a = np.array([r[metric] for r in rows if r['type'] == first], float)
        b = np.array([r[metric] for r in rows if r['type'] == second], float)
        if min(len(a), len(b)) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Independent inference needs two finite observations per group')
        lo, hi = bootstrap_ci_diff(a, b)
        stat, p = ttest_ind(b, a, equal_var=False)
        test = 'two-sided Welch t-test'
        bootstrap = 'independent group percentile bootstrap, seed 0, 10000 draws'
    return {
        'first': first, 'second': second, 'direction': 'second minus first',
        'n_first': len(a), 'n_second': len(b), 'paired_unit': paired_unit,
        'mean_difference': float(b.mean() - a.mean()),
        'mean_difference_ci95': [float(lo), float(hi)],
        'relative_change_percent': float(relative_change_pct(a, b)),
        'cohens_d_pooled_descriptive': float(cohens_d(a, b)),
        'test': test, 'bootstrap': bootstrap, 't': float(stat),
        'p_raw': float(p), 'p_bonferroni': float(min(1.0, p * family_size)),
        'multiplicity': f'Bonferroni over {family_size} pairs within metric; CI unadjusted',
    }


def relative_change_pct(a, b):
    """Percent change of mean from a to b: 100*(mean(b)-mean(a))/|mean(a)|."""
    ma, mb = np.mean(a), np.mean(b)
    if abs(ma) < 1e-12:
        return np.nan
    return 100.0 * (mb - ma) / abs(ma)


def print_effect_table(all_results, metrics, type_order, label_map,
                       pairs, test_name, unit_label, paired_unit=None):
    """
    Print a per-group summary (mean [95% CI], n) followed by a pairwise
    effect-size table: absolute mean difference, CI, descriptive d and p. The caller
    must name the replication unit explicitly.
    """
    print(f"\n{'=' * 78}")
    print(f"  SUMMARY — {test_name}   ({unit_label} is the unit of replication)")
    print(f"{'=' * 78}")

    # Per-group means + bootstrap CI
    cw = 22
    header = f"{'Model':<14}" + "".join(f"{m:>{cw}}" for m in metrics)
    print(header)
    print("-" * len(header))
    for t in type_order:
        rows = [r for r in all_results if r['type'] == t]
        if not rows:
            continue
        line = f"{label_map[t]:<10}(n={len(rows):>2})"
        for m in metrics:
            v = np.array([r[m] for r in rows], float)
            lo, hi = bootstrap_ci_mean(v)
            line += f"{v.mean():.4f}[{lo:.3f},{hi:.3f}]".rjust(cw)
        print(line)
    print("-" * len(header))

    # Pairwise effect sizes
    print(f"\n  Pairwise absolute differences, unadjusted bootstrap 95% CI, "
          f"descriptive pooled d, within-metric Bonferroni p; paired unit={paired_unit}")
    for m in metrics:
        print(f"\n   {m}:")
        for g1, g2 in pairs:
            if paired_unit is not None:
                a, b = paired_values(all_results, m, g1, g2, paired_unit)
            else:
                a = np.array([r[m] for r in all_results if r['type'] == g1], float)
                b = np.array([r[m] for r in all_results if r['type'] == g2], float)
            if len(a) < 2 or len(b) < 2:
                continue
            d = cohens_d(a, b)
            result = pairwise_statistics(all_results, m, g1, g2, paired_unit,
                                          family_size=len(pairs))
            lo, hi = result['mean_difference_ci95']
            pct = relative_change_pct(a, b)
            p = result['p_bonferroni']
            print(f"     {label_map[g1]:>9} → {label_map[g2]:<9}  "
                  f"descriptive d={d:+.2f} "
                  f"delta={result['mean_difference']:+.6f} "
                  f"CI=[{lo:+.6f},{hi:+.6f}]  {pct:+.1f}%  (p_adj={p:.1e})")
    print(f"{'=' * 78}\n")
