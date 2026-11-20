"""Continuum checks for the noise-penalized entropy surrogate."""
import unittest

import numpy as np
from scipy.integrate import quad

from FigS4.noisy_channel_gibbs import C0, C1, LAM_MAX, example_density, gibbs_sweep


class SurrogateQuadratureTests(unittest.TestCase):
    def test_scalar_quadrature_reference(self):
        grid = np.linspace(0, LAM_MAX, 17)
        betas = [0.001, 0.05, 1.0, 10.0 ** 1.5]
        for xi in [0.01, 0.3, 3.0, 30.0]:
            values = gibbs_sweep(xi, betas, grid, grid[1])
            for index, beta in enumerate(betas):
                with self.subTest(xi=xi, beta=beta):
                    sigma0sq = 10 * xi

                    def weight(r):
                        return np.exp(-0.5 * np.log1p(r / sigma0sq) - beta * C1 * r)

                    z = quad(weight, 0, LAM_MAX, epsabs=1e-12, epsrel=1e-12)[0]
                    mean = quad(lambda r: r * weight(r), 0, LAM_MAX,
                                epsabs=1e-12, epsrel=1e-12)[0] / z
                    second = quad(lambda r: r * r * weight(r), 0, LAM_MAX,
                                  epsabs=1e-12, epsrel=1e-12)[0] / z
                    entropy = quad(
                        lambda r: weight(r) / z *
                        (np.log(z) + 0.5 * np.log1p(r / sigma0sq) + beta * C1 * r),
                        0, LAM_MAX, epsabs=1e-12, epsrel=1e-12)[0]
                    penalty = quad(
                        lambda r: weight(r) / z *
                        (0.5 * np.log(2 * np.pi * np.e * (sigma0sq + r))),
                        0, LAM_MAX, epsabs=1e-12, epsrel=1e-12)[0]
                    expected = [mean, np.sqrt(second - mean ** 2) / mean,
                                entropy, entropy - penalty, C0 + C1 * mean]
                    actual = [values[key][index] for key in ["lam", "CV", "H", "I", "C"]]
                    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
                    density = example_density(xi, beta, grid, grid[1])
                    np.testing.assert_allclose(density, weight(grid) / z, rtol=2e-10)

    def test_moments_do_not_depend_on_display_grid(self):
        coarse = np.linspace(0, LAM_MAX, 10)
        fine = np.linspace(0, LAM_MAX, 4000)
        a = gibbs_sweep(30, [0.05, 30], coarse, coarse[1])
        b = gibbs_sweep(30, [0.05, 30], fine, fine[1])
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
