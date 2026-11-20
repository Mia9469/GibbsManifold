#!/usr/bin/env python3
"""
Composite generalization panels for the main text.

The figure compares all three dictionaries (Null, SparseNet, CVM-v2) in both
held-out natural-image and structured-OOD regimes.  Rows are regimes; columns
are the three reported matched-inference metrics: RE, SSIM, and RAE.

Usage:  python make_fig6_generalization_panels.py
Output: generalization_results/Fig6_generalization_panels.{pdf,png}
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
RES_DIR = os.path.join(BASE_DIR, 'generalization_results')
HELDOUT = os.path.join(RES_DIR, 'held_out_natural', 'summary.json')
STRUCT = os.path.join(RES_DIR, 'structured_patterns', 'summary.json')
STRUCT_PAIR = os.path.join(RES_DIR, 'structured_patterns',
                           'pattern_pair_summary.json')
STRUCT_PATTERN = os.path.join(RES_DIR, 'structured_patterns',
                              'summary_by_pattern.json')

PALETTE = {'Null': '#868686FF', 'SparseNet': '#0073C2FF', 'CVM': '#EFC000FF'}
LABEL = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
ORDER = ['Null', 'SparseNet', 'CVM']
TYPE_ORDER = ['null', 'cauchy', 'cv_m']
METRICS = [
    ('RE', 'Reconstruction error (RE)', False),
    ('SSIM', 'Structural similarity (SSIM)', True),
    ('ICE', r'RAE score  $-\log_2(\mathrm{RE})/\bar{a}$', True),
]


def load(path):
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def best_count_note(df, metric, higher, unit_col, prefix="Best"):
    """Return a compact best-unit count note for paired units."""
    counts = {t: 0 for t in TYPE_ORDER}
    units = sorted(df[unit_col].unique())
    for unit in units:
        sub = df[df[unit_col] == unit].set_index('type')
        if not all(t in sub.index for t in TYPE_ORDER):
            continue
        vals = {t: float(sub.loc[t, metric]) for t in TYPE_ORDER}
        best = max(vals, key=vals.get) if higher else min(vals, key=vals.get)
        counts[best] += 1
    parts = [f"{LABEL[t]} {counts[t]}" for t in TYPE_ORDER if counts[t] > 0]
    return f"{prefix}:\n" + ", ".join(parts)


def cvm_vs_count(df, metric, higher, unit_col, comparator):
    """Count paired units where CVM is better than or equal to comparator."""
    n_good = 0
    n_total = 0
    for unit in sorted(df[unit_col].unique()):
        sub = df[df[unit_col] == unit].set_index('type')
        if 'cv_m' not in sub.index or comparator not in sub.index:
            continue
        cvm = float(sub.loc['cv_m', metric])
        comp = float(sub.loc[comparator, metric])
        n_good += int(cvm >= comp if higher else cvm <= comp)
        n_total += 1
    return n_good, n_total


def pattern_notes(pattern_df):
    notes = {}
    for metric, _, higher in METRICS[:2]:
        best = best_count_note(pattern_df, metric, higher, 'pattern',
                               prefix='Pattern best')
        n_null, n_total = cvm_vs_count(pattern_df, metric, higher,
                                       'pattern', 'null')
        notes[metric] = (
            f"CVM better than Null:\n{n_null}/{n_total} patterns\n"
            + best.replace('Pattern best:\n', 'Best: ')
        )
    return notes


def metric_panel(ax, df, metric, title, ylabel, higher, paired_unit=None,
                 note=None):
    """Box/strip panel for one metric, optionally with paired-unit lines."""
    pdf = df.copy()
    pdf['label'] = pdf['type'].map(LABEL)
    if paired_unit is not None:
        pos = {t: i for i, t in enumerate(TYPE_ORDER)}
        for _, sub in pdf.groupby(paired_unit):
            if not all(t in set(sub['type']) for t in TYPE_ORDER):
                continue
            sub = sub.set_index('type')
            y = [float(sub.loc[t, metric]) for t in TYPE_ORDER]
            ax.plot([pos[t] for t in TYPE_ORDER], y, color='0.76',
                    lw=0.8, alpha=0.75, zorder=0)
    sns.boxplot(x='label', y=metric, data=pdf, order=ORDER, hue='label',
                palette=PALETTE, legend=False, width=0.6, fliersize=0, ax=ax)
    sns.stripplot(x='label', y=metric, data=pdf, order=ORDER, color='0.18',
                  size=3.2, alpha=0.65, jitter=0.10, ax=ax, zorder=3)
    means = pdf.groupby('label')[metric].mean().reindex(ORDER)
    ax.scatter(range(len(ORDER)), means.values, marker='D', s=30,
               color='white', edgecolor='k', linewidth=0.7, zorder=4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.set_title(title, fontsize=12, pad=8)
    ax.tick_params(axis='x', rotation=20)
    ax.margins(y=0.25 if note else 0.12)
    if note:
        ax.text(0.03, 0.97, note, transform=ax.transAxes, va='top',
                ha='left', fontsize=8.2,
                bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
    if not higher:
        ax.text(0.98, 0.04, 'lower is better', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='0.35')
    else:
        ax.text(0.98, 0.04, 'higher is better', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='0.35')
    sns.despine(ax=ax)


def method_means(df, metric):
    vals = {}
    for t in TYPE_ORDER:
        vals[t] = float(df.loc[df['type'] == t, metric].mean())
    return vals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=os.path.join(
        BASE_DIR, '..', 'outputs', 'final_text_review_20260907', 'ann_corrected', 'panels'))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    held = load(HELDOUT)
    struct = load(STRUCT)
    pattern_df = load(STRUCT_PATTERN)
    pat_notes = pattern_notes(pattern_df)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), dpi=200,
                             facecolor='w')

    for col, (metric, ylabel, higher) in enumerate(METRICS):
        note = best_count_note(held, metric, higher, 'fold',
                               prefix='Best folds')
        metric_panel(axes[0, col], held, metric,
                     'Held-out natural images (LOO-CV)' if col == 0 else '',
                     ylabel, higher, paired_unit='fold', note=note)

    for col, (metric, ylabel, higher) in enumerate(METRICS):
        note = None
        if metric in pat_notes:
            note = pat_notes[metric]
        elif metric == 'ICE':
            means = method_means(struct, metric)
            note = (f"Run means:\nSparseNet {means['cauchy']:.1f} > "
                    f"CVM {means['cv_m']:.1f} > Null {means['null']:.1f}")
        metric_panel(axes[1, col], struct, metric,
                     'Structured / oriented patterns' if col == 0 else '',
                     ylabel, higher, paired_unit=None, note=note)

    axes[0, 0].set_ylabel('Natural LOO\n' + axes[0, 0].get_ylabel())
    axes[1, 0].set_ylabel('Structured OOD\n' + axes[1, 0].get_ylabel())

    for ax, letter in zip(axes.ravel(), 'abcdef'):
        ax.text(-0.16, 1.08, letter, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')

    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    out = os.path.join(args.output_dir, 'Fig6_generalization_panels')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', bbox_inches='tight', dpi=200)
    print(f"Saved → {out}.pdf / .png")


if __name__ == '__main__':
    main()
