"""Run: python3 -m unittest discover -s FigS4 -p test_sublinear_continuum_audit.py"""

import unittest

import numpy as np

from sublinear_continuum_audit import (
    ALPHAS, BETAS, R_MAX, log_integral, make_records, marker_state,
    state, validate, yield_turn,
)


class ContinuumAuditTests(unittest.TestCase):
    def test_adaptive_quadrature_and_analytic_limits(self):
        result = validate()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["adaptive_quad_cases"]), 24)
        self.assertEqual(result["linear_alpha1_beta5_CV"], 1.0)

    def test_original_grid_and_finite_node_ordering(self):
        records = make_records()
        self.assertEqual(len(records), 8)
        self.assertEqual(len(BETAS), 80)
        for record in records:
            self.assertEqual(len(record["nodes"]), 80)
            self.assertTrue(record["sampled_CV_derivatives_all_positive"])
            self.assertEqual(record["negative_CV_steps_beyond_1e-10_roundoff"], 0)
            span = record["marker_bracketing_nodes"]
            self.assertTrue(span["positive_adjacent_yield_increments"])
            for key in ("min_logZ", "min_dCV_dbeta", "min_dyield_dbeta"):
                self.assertGreater(span[key], 0)

    def test_turning_points_and_marker_roots(self):
        for alpha in ALPHAS:
            turn = yield_turn(alpha)
            self.assertLess(abs(turn["logZ"]), 1e-11)
            self.assertAlmostEqual(turn["entropy_yield"], turn["beta"], places=10)
        self.assertFalse(yield_turn(0.3)["inside_original_beta_sweep"])
        self.assertTrue(yield_turn(0.4)["inside_original_beta_sweep"])
        for cv in (0.63, 0.86):
            self.assertAlmostEqual(marker_state(cv)["CV"], cv, places=11)

    def test_uniform_moments_and_invalid_inputs(self):
        self.assertAlmostEqual(log_integral(0, 0.3, 0), np.log(R_MAX), places=12)
        for alpha, beta in ((0, 1), (-1, 1), (0.7, -1), (np.nan, 1)):
            with self.assertRaises(ValueError):
                state(alpha, beta)


if __name__ == "__main__":
    unittest.main()
