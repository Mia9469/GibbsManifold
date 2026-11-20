#!/usr/bin/env python3
"""Checkpoint-only ANN metric correction. No training and no manuscript writes.

Stage all numerical products under the requested audit output directory; publish
only validated Fig6 JSON/CSV/NPZ inputs after preserving any existing originals.
"""

import argparse
import copy
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.stats import ttest_ind
from skimage.metrics import structural_similarity

from evaluation_metrics import (SSIM_PROTOCOL, exact_unique_inputs, reference_ssim,
                                validate_frozen_manifest)
from generalization_test import (MODEL_DIRS, evaluate_dictionary,
                                 generate_structured_patterns,
                                 generate_pink_noise_dataset)
from heldout_generalization_test import extract_test_patches, METHODS
from gen_stats import pairwise_statistics
from p3_matched import p3_eval


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/final_text_review_20260907/ann_corrected'
BACKUP = ROOT / 'backups/pre_final_text_audit_20260907'
LABELS = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
PAIRS = [('null', 'cauchy'), ('null', 'cv_m'), ('cauchy', 'cv_m')]
VERSION = '20260907-reference-range-exact-dedup-v1'


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value):
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
    encoded = json.dumps(value, indent=2, allow_nan=False, default=scalar) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(encoded)
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def original(relative):
    saved = BACKUP / relative
    return saved if saved.exists() else ROOT / relative


def stage(relative):
    return OUT / relative


def save(relative, value):
    write_json(stage(relative), value)


def baseline_check(old_rows, new_rows, keys, metrics, tolerance=3e-6):
    before = {tuple(r[k] for k in keys): r for r in old_rows}
    after = {tuple(r[k] for k in keys): r for r in new_rows}
    if len(before) != len(old_rows) or len(after) != len(new_rows) or set(before) != set(after):
        raise ValueError('Frozen cohort identities differ')
    differences = {
        metric: max(abs(float(before[k][metric]) - float(after[k][metric])) for k in before)
        for metric in metrics
    }
    if max(differences.values()) > tolerance:
        raise ValueError(f'Non-SSIM frozen reproduction failed: {differences}')
    return differences


def evaluate_saved(path, patches, cache_key, per_patch=False):
    """Strict, resumable evaluation of a saved dictionary, with input hashes."""
    if not path.is_file():
        raise FileNotFoundError(f'No retraining permitted; missing {path}')
    code_hash = hashlib.sha256(''.join(sha256(Path(__file__).parent / p) for p in
        ('generalization_test.py', 'evaluation_metrics.py')).encode()).hexdigest()
    provenance = {
        'version': VERSION, 'code_sha256': code_hash,
        'checkpoint': str(path.relative_to(ROOT)), 'checkpoint_sha256': sha256(path),
        'input_sha256': hashlib.sha256(np.ascontiguousarray(patches).tobytes()).hexdigest(),
        'return_per_patch': per_patch,
    }
    cache = OUT / 'checkpoint_cache' / (cache_key + '.json')
    if cache.exists():
        try:
            previous = read_json(cache)
        except json.JSONDecodeError:
            previous = {}
        if previous.get('provenance') == provenance:
            return previous['result']
    net = torch.load(path, map_location='cpu', weights_only=False)
    u = net.U.weight.detach().numpy()
    if u.shape != (256, 144) or not np.isfinite(u).all():
        raise ValueError(f'Invalid saved dictionary: {path}')
    # Apple's BLAS can leave spurious floating status flags; evaluate_dictionary
    # explicitly rejects non-finite weights, coefficients, reconstructions and metrics.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        result = evaluate_dictionary(u, patches, return_per_patch=per_patch)
    if result['converged_rate'] != 1:
        raise ValueError(f'Frozen solver failed its original stopping criterion: {path}')
    write_json(cache, {'provenance': provenance, 'result': result})
    return result


def summarize(rows, unit, paired_unit=None):
    metrics = ['RE', 'SSIM', 'ICE', 'mean_a', 'independence']
    summary = {'unit': unit, 'SSIM_protocol': SSIM_PROTOCOL, 'groups': {}, 'pairwise': {}}
    for typ, label in LABELS.items():
        selected = [r for r in rows if r['type'] == typ]
        summary['groups'][label] = {'n': len(selected)}
        for metric in metrics:
            values = np.asarray([r[metric] for r in selected])
            summary['groups'][label][metric] = {
                'mean': float(values.mean()), 'sd': float(values.std(ddof=1)),
            }
    for first, second in PAIRS:
        summary['pairwise'][f'{LABELS[first]} vs {LABELS[second]}'] = {
            metric: pairwise_statistics(rows, metric, first, second, paired_unit)
            for metric in metrics
        }
    summary['inference_scope'] = (
        'Paired image-fold contrasts, conditional on the ten-image dataset; LOO training sets overlap'
        if paired_unit else 'Initialization variability conditional on the fixed input bank; not stimulus-population inference'
    )
    summary['aggregation'] = {
        'RE': 'Mean patch norm ratio within checkpoint; checkpoint means within run, then seeds within fold for LOO',
        'ICE': 'Legacy RAE key: -log2(checkpoint mean RE) / checkpoint mean absolute activation, then checkpoint/run/fold means',
        'SSIM': 'Mean patch SSIM with each reference patch range; same checkpoint/run/fold averaging',
        'cv_pop': 'Legacy generalization diagnostic uses per-patch population SD (ddof=0), mean stabilizer 1e-12; not the training CV statistic',
        'independence': 'Legacy key: mean 1 minus absolute atom-pixel Pearson correlation, percent',
    }
    return summary


def training():
    image_path = ROOT / 'Fig6/src/model/IMAGES.mat'
    images = loadmat(image_path)['IMAGES']
    input_hash = sha256(image_path)
    code_hash = hashlib.sha256((inspect.getsource(training) +
        sha256(Path(__file__).parent / 'evaluation_metrics.py')).encode()).hexdigest()
    rows, checkpoints = [], []
    metrics = ['RE', 'SSIM', 'cost', 'independence', 'CV', 'CV_lt']
    for typ, directory in MODEL_DIRS.items():
        for run in range(20):
            run_checkpoints = []
            for image in range(10):
                path = Path(directory) / f'ckpt-10-{run}-{image+1}.pth'
                cache = OUT / f'checkpoint_cache/training/{typ}-{run}-{image}.json'
                provenance = {'version': VERSION, 'checkpoint_sha256': sha256(path),
                              'checkpoint': str(path.relative_to(ROOT)),
                              'input_sha256': input_hash, 'code_sha256': code_hash}
                if cache.exists() and read_json(cache)['provenance'] == provenance:
                    row = read_json(cache)['result']
                else:
                    net = torch.load(path, map_location='cpu', weights_only=False)
                    weights = net.U.weight.detach().numpy()
                    coefficients = net.R.detach().abs().numpy()
                    if coefficients.shape != (1024, 144) or not np.isfinite(coefficients).all():
                        raise ValueError(f'Invalid cached coefficients: {path}')
                    reconstruction = net.U(net.R).detach().numpy()
                    restored = np.zeros((512, 512))
                    for i, patch in enumerate(reconstruction):
                        x, y = i % 32 * 16, i // 32 * 16
                        restored[x:x+16, y:y+16] = patch.reshape(16, 16)
                    reference = images[:, :, image]
                    dec = 1 - np.abs(np.corrcoef(weights, rowvar=False))
                    row = {
                        'type': typ, 'run': run, 'image': image, 'checkpoint': str(path),
                        'RE': float(np.linalg.norm(reference-restored) / (np.linalg.norm(reference)+1e-12)),
                        'SSIM': reference_ssim(reference, restored),
                        'legacy_SSIM': float(structural_similarity(reference, restored,
                            data_range=restored.max()-restored.min())),
                        'cost': float(coefficients.mean(axis=0).mean()),
                        'independence': float(dec.sum() / (144*143) * 100),
                        'CV': float(np.mean(coefficients.std(axis=1, ddof=1) /
                                            (coefficients.mean(axis=1)+1e-8))),
                        'CV_lt': float(np.mean(coefficients.std(axis=0, ddof=1) /
                                               (coefficients.mean(axis=0)+1e-8))),
                        'constant_reference': bool(reference.max() == reference.min()),
                    }
                    write_json(cache, {'provenance': provenance, 'result': row})
                run_checkpoints.append(row)
            checkpoints.extend(run_checkpoints)
            row = {m: float(np.mean([r[m] for r in run_checkpoints])) for m in metrics+['legacy_SSIM']}
            row.update(type=typ, run=run, ICE=float(-np.log2(row['RE']+1e-12)/(row['cost']+1e-12)))
            rows.append(row)
            if (run+1) % 5 == 0:
                print(f'Training: {typ} {run+1}/20 complete', flush=True)
    stats = {
        'unit_of_replication': 'independent training run', 'test': 'two-sided Welch t-test',
        'multiplicity': 'Bonferroni correction over three pairwise model comparisons within each metric',
        'metric_protocol': SSIM_PROTOCOL,
        'aggregation': {
            'training': 'Ten sequential epoch-10 image checkpoints averaged within each run',
            'SSIM': 'Whole training image, reference-only range; cached constraint-aware coefficients',
            'CV': 'Endpoint coefficient sample SD over 144 atoms, divided by mean plus 1e-8, averaged over patches',
            'CV_lt': 'Endpoint coefficient sample SD over patches per atom, divided by mean plus 1e-8, averaged over atoms',
            'RE': 'Mean whole-image norm ratio within run',
            'ICE': 'Legacy RAE name: -log2(run mean RE) divided by run mean absolute activation',
            'independence': 'Legacy name: mean 1 minus absolute atom-pixel Pearson correlation, percent',
        }, 'groups': {}, 'pairwise': {},
    }
    for typ, label in LABELS.items():
        group = [r for r in rows if r['type'] == typ]
        stats['groups'][label] = {'n': 20}
        for metric in metrics+['ICE']:
            values = [r[metric] for r in group]
            stats['groups'][label][metric] = {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))}
    for first, second in PAIRS:
        comparison = {}
        for metric in metrics+['ICE']:
            a = [r[metric] for r in rows if r['type'] == first]
            b = [r[metric] for r in rows if r['type'] == second]
            t, p = ttest_ind(a, b, equal_var=False)
            comparison[metric] = {'t': float(t), 'p_raw': float(p), 'p_bonferroni': float(min(1., p*3))}
        stats['pairwise'][f'{LABELS[first]} vs {LABELS[second]}'] = comparison
    old = read_json(original('Fig6/sparse_run_statistics.json'))
    for typ, label in LABELS.items():
        legacy = np.mean([r['legacy_SSIM'] for r in rows if r['type'] == typ])
        if abs(legacy - old['groups'][label]['SSIM']['mean']) > 1e-9:
            raise ValueError('Legacy training SSIM not reproduced from saved reconstruction')
    stats['legacy_SSIM_reproduction_max_difference'] = max(abs(
        np.mean([r['legacy_SSIM'] for r in rows if r['type'] == typ]) - old['groups'][label]['SSIM']['mean'])
        for typ, label in LABELS.items())
    save('Fig6/sparse_run_statistics.json', stats)
    save('Fig6/sparse_run_rows.json', rows)
    save('Fig6/sparse_checkpoint_rows.json', checkpoints)


def heldout():
    images = loadmat(ROOT / 'Fig6/src/model/IMAGES.mat')['IMAGES']
    rows = []
    for fold in range(10):
        patches = extract_test_patches(images[:, :, fold:fold+1])
        for typ, _, model_key in METHODS:
            for run in range(6):
                path = ROOT / f'Fig6/record_heldout/fold{fold}/trained_models_{model_key}/ckpt-{run}.pth'
                result = evaluate_saved(path, patches, f'heldout/{typ}-{fold}-{run}')
                rows.append(dict(result, type=typ, fold=fold, run=run))
        print(f'Held-out image {fold+1}/10 complete', flush=True)
    old = read_json(original('Fig6/generalization_results/held_out_natural/summary_per_run.json'))
    reproduced = baseline_check(old, rows, ['type', 'fold', 'run'],
                                ['RE', 'ICE', 'mean_a', 'independence'])
    fields = ['RE', 'SSIM', 'ICE', 'mean_a', 'independence']
    folds = pd.DataFrame(rows).groupby(['type', 'fold'])[fields].mean().reset_index().to_dict('records')
    base = 'Fig6/generalization_results/held_out_natural/'
    save(base + 'summary_per_run.json', rows)
    save(base + 'summary.json', folds)
    stats = summarize(folds, 'held-out image fold', paired_unit='fold')
    stats['frozen_non_SSIM_max_differences'] = reproduced
    stats['n_dictionary_checkpoints'] = len(rows)
    stats['n_constant_references'] = sum(r['n_constant_references'] for r in rows)
    save(base + 'statistics.json', stats)


def aggregate_unique(result, indices):
    patches = [result['per_patch'][int(i)] for i in indices]
    re = np.asarray([r['RE'] for r in patches])
    ma = float(np.mean([r['mean_a'] for r in patches]))
    output = {k: v for k, v in result.items() if k != 'per_patch'}
    output.update(RE=float(re.mean()), RE_std=float(re.std()),
                  SSIM=float(np.mean([r['SSIM'] for r in patches])), mean_a=ma,
                  ICE=float(-np.log2(re.mean() + 1e-12) / (ma + 1e-12)),
                  cv_pop=float(np.mean([r['cv_pop'] for r in patches])),
                  n_scored_inputs=len(indices))
    return output, patches


def structured_or_pink(structured):
    kind = 'structured_patterns' if structured else 'pink_noise'
    indices, manifest = None, None
    if structured:
        bank = generate_structured_patterns(patch_size=16, seed=42)
        names = sorted(bank)
        patches = np.stack([bank[n] for n in names])
        indices, manifest = exact_unique_inputs(names, patches)
        if len(indices) != 30 or manifest['aliases'] != {'edge_90deg': 'edge_45deg'}:
            raise ValueError('Unexpected historical input bank; refusing to replace experiment')
        base = f'Fig6/generalization_results/{kind}/'
        canonical_manifest = ROOT / (base + 'input_manifest.json')
        if canonical_manifest.exists():
            validate_frozen_manifest(manifest, read_json(canonical_manifest))
        save(base + 'input_manifest.json', manifest)
        np.savez_compressed(stage(base + 'frozen_input_bank.npz'),
                            original_inputs=patches, original_names=names,
                            retained_indices=indices, unique_inputs=patches[indices],
                            unique_names=manifest['retained_names'])
    else:
        _, patches = generate_pink_noise_dataset(n_images=10, img_size=128, seed=42)
    rows, original_rows, per_run_pattern = [], [], []
    model_dirs = dict(MODEL_DIRS)
    if not structured:
        # Frozen pink-noise results predate the v2 CVM cohort. Preserve that
        # experiment and require non-SSIM reproduction below before publication.
        model_dirs['cv_m'] = ROOT / 'Fig6/record/trained_models_cv_con_new'
    for typ, directory in model_dirs.items():
        for run in range(20):
            checkpoints, original_checkpoints, pattern_checkpoints = [], [], []
            for checkpoint in range(1, 11):
                path = Path(directory) / f'ckpt-10-{run}-{checkpoint}.pth'
                result = evaluate_saved(path, patches, f'{kind}/{typ}-{run}-{checkpoint}',
                                        per_patch=structured)
                original_checkpoints.append({k: v for k, v in result.items() if k != 'per_patch'})
                if structured:
                    scored, pattern_results = aggregate_unique(result, indices)
                    checkpoints.append(scored)
                    pattern_checkpoints.append(pattern_results)
                else:
                    checkpoints.append(result)
            for destination, values in ((rows, checkpoints), (original_rows, original_checkpoints)):
                averaged = {key: float(np.mean([v[key] for v in values])) for key in values[0]}
                destination.append(dict(averaged, type=typ, run=run,
                                        n_checkpoints=10, n_skipped_checkpoints=0))
            if structured:
                for i, name in enumerate(manifest['retained_names']):
                    values = {m: float(np.mean([c[i][m] for c in pattern_checkpoints]))
                              for m in ['RE', 'SSIM', 'mean_a']}
                    per_run_pattern.append(dict(values, type=typ, run=run, pattern=name))
            if (run+1) % 5 == 0:
                print(f'{kind}: {typ} {run+1}/20 runs complete', flush=True)
    old = read_json(original(f'Fig6/generalization_results/{kind}/summary.json'))
    reproduced = baseline_check(old, original_rows, ['type', 'run'],
                                ['RE', 'ICE', 'mean_a', 'independence'])
    base = f'Fig6/generalization_results/{kind}/'
    save(base + 'summary.json', rows)
    stats = summarize(rows, 'training run')
    stats['frozen_non_SSIM_max_differences_before_deduplication'] = reproduced
    stats['n_dictionary_checkpoints'] = 600
    stats['checkpoint_directories'] = {k: str(Path(v).relative_to(ROOT)) for k, v in model_dirs.items()}
    stats['CVM_checkpoint_family'] = 'current v2' if structured else 'historical cv_con; not the Figure7 v2 cohort'
    stats['n_constant_references'] = sum(r['n_constant_references'] * 10 for r in rows)
    if structured:
        fields = ['RE', 'SSIM', 'mean_a']
        pattern_rows = pd.DataFrame(per_run_pattern).groupby(['type', 'pattern'])[fields].mean().reset_index().to_dict('records')
        save(base + 'summary_per_run_pattern.json', per_run_pattern)
        save(base + 'summary_by_pattern.json', pattern_rows)
        pair = {'metric': 'SSIM', 'unit': 'unique predefined structured input',
                'n_patterns': 30, 'n_original_entries': 31, 'aliases': manifest['aliases'],
                'inference_scope': 'Descriptive counts in a fixed bank, not independent stimulus-population samples',
                'metrics': {}}
        for metric, higher in [('RE', False), ('SSIM', True)]:
            pivot = pd.DataFrame(pattern_rows).pivot(index='pattern', columns='type', values=metric)
            diff = pivot.cv_m - pivot['null']
            counts = {key: int(((pivot[key] >= pivot.max(axis=1)) if higher else
                              (pivot[key] <= pivot.min(axis=1))).sum()) for key in LABELS}
            entry = {'n_patterns': 30, 'cvm_better_than_null': int(((diff > 0) if higher else (diff < 0)).sum()),
                     'best_counts': counts, 'mean_diff': float(diff.mean()),
                     'median_diff': float(diff.median()), 'min_diff': float(diff.min()),
                     'max_diff': float(diff.max())}
            pair['metrics'][metric] = entry
            if metric == 'SSIM':
                pair.update({k: entry[k] for k in ['mean_diff', 'median_diff', 'min_diff', 'max_diff']})
                pair.update(cvm_gt_null=int((diff > 0).sum()), cvm_ge_null=int((diff >= 0).sum()))
        save(base + 'pattern_pair_summary.json', pair)
        stats['input_manifest'] = manifest
    save(base + 'statistics.json', stats)


def matching():
    rel = 'Fig6/p3_matched/results/eval/'
    with open(original(rel + 'p3_summary.csv'), newline='') as f:
        raw = list(csv.DictReader(f))
    summary = [{k: (v if k in {'method', 'submethod', 'knob_name'} else
                    int(v) if k == 'n_seeds' else float(v)) for k, v in r.items()}
               for r in raw if r['method'] in {'null', 'cauchy', 'cvmax_v2'}]
    if len(summary) != 32 or sum(r['n_seeds'] for r in summary) != 96:
        raise ValueError('Wrong current-v2 frozen sweep cohort')
    directory = stage(rel)
    directory.mkdir(parents=True, exist_ok=True)
    p3_eval.write_csv(directory / 'p3_summary.csv', summary)
    with open(original(rel + 'p3_long.csv'), newline='') as f:
        long_rows = [r for r in csv.DictReader(f) if r['method'] in {'null', 'cauchy', 'cvmax_v2'}]
    if len(long_rows) != 96:
        raise ValueError('Wrong long-form sweep count')
    p3_eval.write_csv(directory / 'p3_long.csv', long_rows)
    for axis, target, label, filename in [
        ('RE', np.array([.60, .62, .65]), 'matched-RE', 'p3_matched_RE'),
        ('mean_a', np.array([.045, .049, .055, .058]), 'matched-a', 'p3_matched_a'),
    ]:
        for outcome, suffix, better in [('mean_a' if axis == 'RE' else 'RE', '', 'min'),
                                         ('ICE_log_a', '_ICE', 'max')]:
            rows = p3_eval.matched_table(summary, axis, outcome, target, label, better)
            p3_eval.write_csv(directory / (filename + suffix + '.csv'), rows)
    save(rel + 'matching_provenance.json', p3_eval.matching_provenance(summary))
    figures = OUT / 'panels'
    figures.mkdir(parents=True, exist_ok=True)
    p3_eval.plot_pareto(summary, figures / 'matched_operating_curves.pdf')
    p3_eval.plot_ice_cv(summary, 'ICE_log_a', figures / 'matched_RAE_CV.pdf')
    p3_eval.plot_ice_vs_re(summary, 'ICE_log_a', figures / 'matched_RAE_RE.pdf')


def historical_validation():
    old = read_json(original('Fig6/seeded_rerun_validation.json'))
    new = copy.deepcopy(old)
    new['legacy_audit'] = old
    new['purpose'] = 'Historical deterministic audit; unavailable checkpoints prevent validation of corrected SSIM and endpoint CV'
    new['correction_scope'] = {
        'version': VERSION, 'validated_corrected_SSIM': False,
        'validated_endpoint_CV': False,
        'reason': 'Seeded-rerun checkpoints were previously deleted; no retraining permitted',
        'legacy_values': 'Preserved unchanged under legacy_audit and in the pre-correction backup',
    }
    for group in new['groups'].values():
        for metric in ['SSIM_mean', 'CV_population_mean', 'CV_lifetime_mean']:
            group[metric] = None
    for key in ['all_metric_rankings_unchanged', 'maximum_absolute_group_mean_shift',
                'maximum_shift_metric', 'maximum_bonferroni_adjusted_pairwise_p_across_metrics']:
        new['comparison_to_archival_summary'][key] = None
    new['comparison_to_archival_summary']['interpretation'] = 'Historical inference only; not a validation of the corrected metric cohort'
    save('Fig6/seeded_rerun_validation.json', new)


def verify():
    train = read_json(stage('Fig6/sparse_run_statistics.json'))
    old_train = read_json(original('Fig6/sparse_run_statistics.json'))
    rows = read_json(stage('Fig6/sparse_run_rows.json'))
    checkpoints = read_json(stage('Fig6/sparse_checkpoint_rows.json'))
    if len(rows) != 60 or len(checkpoints) != 600:
        raise ValueError('Training cohort incomplete')
    training_differences = {}
    s6 = read_json(ROOT / 'outputs/supp_revision_20260904/S6/FigureS6_v2_statistics.json')
    for method, group in train['groups'].items():
        if group['n'] != 20:
            raise ValueError('Training run count changed')
        for metric in ['RE', 'ICE', 'cost', 'independence']:
            difference = abs(group[metric]['mean'] - old_train['groups'][method][metric]['mean'])
            training_differences[f'{method}/{metric}'] = difference
            # The checkpoint-only path averages float32 costs in float64;
            # retain tight tolerances for that reduction-order difference.
            tolerance = {'cost': 1e-8, 'ICE': 3e-7}.get(metric, 1e-9)
            if difference > tolerance:
                raise ValueError(f'Unintended training metric drift: {method} {metric}')
        for metric, s6_metric in [('CV', 'CV_P'), ('CV_lt', 'CV_L'), ('cost', 'mean_activation')]:
            for stat in ['mean', 'sd']:
                if abs(group[metric][stat] - s6['groups'][method][s6_metric][stat]) > 3e-7:
                    raise ValueError(f'Endpoint metric disagrees with S6: {method}/{metric}/{stat}')
        typ = next(k for k, label in LABELS.items() if label == method)
        selected = [r for r in rows if r['type'] == typ]
        if {r['run'] for r in selected} != set(range(20)):
            raise ValueError('Missing training run identity')
        for metric in ['RE', 'SSIM', 'ICE', 'cost', 'independence', 'CV', 'CV_lt']:
            if abs(np.mean([r[metric] for r in selected]) - group[metric]['mean']) > 1e-12:
                raise ValueError('Training row/group mismatch')
    reports = {}
    for kind in ['held_out_natural', 'structured_patterns', 'pink_noise']:
        base = f'Fig6/generalization_results/{kind}/'
        reports[kind] = read_json(stage(base + 'statistics.json'))
        expected = 10 if kind == 'held_out_natural' else 20
        if any(g['n'] != expected for g in reports[kind]['groups'].values()):
            raise ValueError(f'Wrong group count for {kind}')
    base = 'Fig6/generalization_results/structured_patterns/'
    manifest = read_json(stage(base + 'input_manifest.json'))
    with np.load(stage(base + 'frozen_input_bank.npz')) as bank:
        indices, observed = exact_unique_inputs(list(bank['original_names']), bank['original_inputs'])
        if observed != manifest or len(indices) != 30 or not np.array_equal(
                bank['unique_inputs'], bank['original_inputs'][indices]):
            raise ValueError('Frozen structured bank and manifest disagree')
    pattern_rows = read_json(stage(base + 'summary_by_pattern.json'))
    run_patterns = read_json(stage(base + 'summary_per_run_pattern.json'))
    if len(pattern_rows) != 90 or len(run_patterns) != 1800:
        raise ValueError('Structured pattern cohort incomplete')
    for typ, label in LABELS.items():
        selected = [r for r in pattern_rows if r['type'] == typ]
        if {r['pattern'] for r in selected} != set(manifest['retained_names']):
            raise ValueError('Structured pattern identities differ')
        for metric in ['RE', 'SSIM', 'mean_a']:
            if abs(np.mean([r[metric] for r in selected]) - reports['structured_patterns']['groups'][label][metric]['mean']) > 1e-12:
                raise ValueError('Structured pattern means and run summaries disagree')
    source_inputs = {}
    for path in sorted(stage('Fig6').rglob('*')):
        if path.is_file() and path.suffix in {'.json', '.csv', '.npz'}:
            source_inputs[str(path.relative_to(OUT))] = sha256(path)
    report = {'version': VERSION, 'training': train, 'regimes': reports,
              'structured_pattern_comparison': read_json(stage('Fig6/generalization_results/structured_patterns/pattern_pair_summary.json')),
              'SSIM_protocol': SSIM_PROTOCOL, 'source_inputs_sha256': source_inputs,
              'training_non_SSIM_max_differences': training_differences,
              'endpoint_S6_agreement_tolerance': 3e-7,
              'retraining': False, 'artwork_overwritten': False,
              'training_constant_references': sum(r['constant_reference'] for r in checkpoints)}
    write_json(OUT / 'numerical_summary.json', report)
    for regime, groups in [('training', train['groups'])] + [(k, v['groups']) for k, v in reports.items()]:
        print(regime, json.dumps(groups), flush=True)
    return report


def publish():
    report = verify()
    published = []
    for relative, expected_hash in report['source_inputs_sha256'].items():
        path, target = stage(relative), ROOT / relative
        if sha256(path) != expected_hash or not relative.startswith('Fig6/'):
            raise ValueError('Publication manifest mismatch')
        if target.exists():
            backup = BACKUP / relative
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256(target) != expected_hash:
            raise ValueError('Publication checksum mismatch')
        published.append(relative)
    write_json(OUT / 'publication_manifest.json', {
        'version': VERSION, 'files': published, 'sha256': report['source_inputs_sha256'],
        'shared_workbook_builder_modified': False, 'manuscript_modified': False,
        'note': 'Current SourceData builder reads these canonical Fig6 inputs. Rebuild the workbook in the parent task.',
    })
    print(f'Published {len(published)} validated numerical inputs; artwork untouched', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', default='training,heldout,structured,pink,matching,validation,verify')
    args = parser.parse_args()
    torch.set_num_threads(1)
    actions = {'training': training, 'heldout': heldout, 'structured': lambda: structured_or_pink(True),
               'pink': lambda: structured_or_pink(False), 'matching': matching,
               'validation': historical_validation, 'verify': verify, 'publish': publish}
    timings = {}
    for name in args.stages.split(','):
        start = time.perf_counter()
        print(f'START {name}', flush=True)
        actions[name]()
        timings[name] = time.perf_counter() - start
        print(f'COMPLETE {name}: {timings[name]:.1f} seconds', flush=True)
    write_json(OUT / 'latest_stage_timings.json', timings)


if __name__ == '__main__':
    main()
