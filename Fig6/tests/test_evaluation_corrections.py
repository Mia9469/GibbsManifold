import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_metrics import (reference_range, reference_ssim, exact_unique_inputs,
                                validate_frozen_manifest)
from gen_stats import paired_values, pairwise_statistics
from generalization_test import generate_structured_patterns, evaluate_dictionary
from p3_matched.p3_eval import interp_at, matching_provenance


class EvaluationCorrections(unittest.TestCase):
    def test_ssim_reference_range(self):
        ref = np.arange(256, dtype=float).reshape(16, 16) / 255
        for scale in (0.1, 1, 10):
            rec = scale * ref
            self.assertAlmostEqual(reference_ssim(ref, rec),
                                   structural_similarity(ref, rec, data_range=1.0))
        self.assertEqual(reference_range(ref), 1.0)

    def test_constant_reference(self):
        ref = np.zeros((16, 16))
        self.assertEqual(reference_range(ref), 1.0)
        self.assertEqual(reference_ssim(ref, ref), 1.0)
        self.assertTrue(np.isfinite(reference_ssim(ref, np.ones_like(ref))))

    def test_nonfinite_rejected(self):
        ref = np.zeros((16, 16))
        with self.assertRaises(ValueError):
            reference_ssim(ref, np.full_like(ref, np.nan))

    def test_exact_frozen_duplicates(self):
        patterns = generate_structured_patterns()
        names = sorted(patterns)
        patches = np.stack([patterns[n] for n in names])
        keep, manifest = exact_unique_inputs(names, patches)
        self.assertEqual(len(keep), 30)
        self.assertEqual(manifest['aliases'], {'edge_90deg': 'edge_45deg'})
        for name in manifest['retained_names']:
            self.assertTrue(np.array_equal(patterns[name], patches[names.index(name)]))
        almost = patches[:1].copy()
        almost[0, 0] = np.nextafter(almost[0, 0], np.inf)
        indices, _ = exact_unique_inputs(['original', 'different'],
                                         np.vstack([patches[0], almost[0]]))
        self.assertEqual(len(indices), 2)

    def test_frozen_guard_rejects_changed_unique_input(self):
        patterns = generate_structured_patterns()
        names = sorted(patterns)
        patches = np.stack([patterns[n] for n in names])
        _, frozen = exact_unique_inputs(names, patches)
        validate_frozen_manifest(frozen, frozen)
        patches[0, 0] = np.nextafter(patches[0, 0], np.inf)
        _, changed = exact_unique_inputs(names, patches)
        self.assertEqual(changed['n_unique_inputs'], 30)
        with self.assertRaises(ValueError):
            validate_frozen_manifest(changed, frozen)

    def test_dedup_scores_without_changing_inference(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(7)
        u = rng.normal(size=(256, 8)).astype('float32')
        u /= np.linalg.norm(u, axis=0)
        patches = rng.normal(size=(3, 256)).astype('float32')
        patches[2] = patches[1]
        full = evaluate_dictionary(u, patches, return_per_patch=True)
        unique = evaluate_dictionary(u, patches, return_per_patch=True, scoring_indices=[0, 1])
        self.assertEqual(full['avg_iters'], unique['avg_iters'])
        self.assertEqual(unique['n_inference_inputs'], 3)
        self.assertEqual(unique['n_scored_inputs'], 2)
        self.assertEqual(full['per_patch'][:2], unique['per_patch'])

    def test_pairing_is_by_fold_not_order(self):
        rows = [dict(type='a', fold=i, RE=100 * i) for i in range(5)]
        rows += [dict(type='b', fold=i, RE=100 * i + i + 1) for i in reversed(range(5))]
        a, b = paired_values(rows, 'RE', 'a', 'b', 'fold')
        np.testing.assert_array_equal(b - a, np.arange(1, 6))
        result = pairwise_statistics(rows, 'RE', 'a', 'b', 'fold')
        self.assertEqual(result['mean_difference'], 3)
        self.assertGreater(result['mean_difference_ci95'][0], 0)
        with self.assertRaises(ValueError):
            paired_values(rows + [rows[0]], 'RE', 'a', 'b', 'fold')
        with self.assertRaises(ValueError):
            paired_values(rows[:-1], 'RE', 'a', 'b', 'fold')

    def test_matching_no_extrapolation_and_provenance(self):
        values = interp_at(np.array([0.6, 0.62]), np.array([12., 14.]),
                           np.array([0.59, 0.61, 0.63]))
        self.assertTrue(np.isnan(values[0]) and np.isnan(values[2]))
        self.assertAlmostEqual(values[1], 13)
        meta = matching_provenance([dict(n_seeds=3)])
        self.assertIn('Direct linear', meta['interpolation'])
        self.assertEqual(meta['n_runs'], 3)


if __name__ == '__main__':
    unittest.main()
