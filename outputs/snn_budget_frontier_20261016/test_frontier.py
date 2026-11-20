"""Numerical checks against independent constrained optimization and moments."""
import unittest

import numpy as np
from scipy.optimize import minimize

from frontier import entropy, gibbs, metrics, thin


class FrontierTests(unittest.TestCase):
    def test_matches_independent_entropy_optimizer(self):
        maximum, mean = 12, 2.5
        k = np.arange(maximum + 1)
        q, _ = gibbs(mean, maximum)
        result = minimize(lambda p: -entropy(p), np.full(maximum + 1, 1 / (maximum + 1)),
                          bounds=[(1e-12, 1)] * (maximum + 1),
                          constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1},
                                       {"type": "eq", "fun": lambda p: p @ k - mean}],
                          method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, q, atol=2e-6)
        self.assertAlmostEqual(metrics(q)["deficit_nats"], 0., places=12)

    def test_same_cv_does_not_identify_optimality(self):
        q, _ = gibbs(2., 20)
        p = q.copy()
        p[:4] += .01 * np.array([-1, 3, -3, 1])
        self.assertAlmostEqual(metrics(p)["mean_count"], metrics(q)["mean_count"], places=12)
        self.assertAlmostEqual(metrics(p)["cv_pooled"], metrics(q)["cv_pooled"], places=12)
        self.assertGreater(metrics(p)["deficit_nats"], .001)

    def test_thinning_preserves_the_correct_budget(self):
        p = np.array([.1, .2, .3, .4])
        q = thin(p, .25)
        np.testing.assert_allclose(q @ np.arange(4), .25 * (p @ np.arange(4)), atol=1e-12)
        np.testing.assert_allclose(thin(p, 0), [1, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(thin(p, 1), p, atol=1e-12)

    def test_rejects_incompatible_budgets_and_bad_data(self):
        with self.assertRaises(ValueError):
            gibbs(6.1, 12)
        with self.assertRaises(ValueError):
            metrics([.8, .8])


if __name__ == "__main__":
    unittest.main()
