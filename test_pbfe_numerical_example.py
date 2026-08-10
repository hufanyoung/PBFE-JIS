"""Unit checks for the reduced PBFE numerical illustration."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.stats import lognorm

import pbfe_numerical_example as pbfe


class CorrectedNumericalExampleTests(unittest.TestCase):
    def test_lognormal_conversion_recovers_arithmetic_moments(self) -> None:
        means = np.array([0.25, 0.75, 1.0, 1.8])
        for cov in (0.2, 0.5):
            mu_ln, sigma_ln = pbfe.lognormal_parameters_from_arithmetic_mean_cov(
                means, cov
            )
            recovered_means = lognorm.mean(
                s=sigma_ln, loc=0.0, scale=np.exp(mu_ln)
            )
            recovered_stds = lognorm.std(
                s=sigma_ln, loc=0.0, scale=np.exp(mu_ln)
            )
            np.testing.assert_allclose(recovered_means, means, rtol=1e-12)
            np.testing.assert_allclose(recovered_stds / recovered_means, cov, rtol=1e-12)

    def test_finite_domain_masses_are_normalized(self) -> None:
        results = pbfe.calculate_reduced_example()
        np.testing.assert_allclose(results["lai_masses"].sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(results["tfp_masses"].sum(axis=1), 1.0, atol=1e-12)

    def test_loss_means_are_positive_on_tfp_domain(self) -> None:
        results = pbfe.calculate_reduced_example()
        self.assertGreater(float(np.min(results["conditional_loss_means"])), 0.0)
        self.assertLess(float(np.max(results["tfp_centers"])), 2.0)

    def test_final_exceedance_is_bounded_and_monotone(self) -> None:
        results = pbfe.calculate_reduced_example()
        curve = results["loss_exceedance"]
        self.assertGreaterEqual(float(np.min(curve)), -1e-12)
        self.assertLessEqual(float(np.max(curve)), 1.0 + 1e-12)
        self.assertTrue(np.all(np.diff(curve) <= 1e-12))
        self.assertAlmostEqual(float(curve[0]), 1.0, places=12)

    def test_expected_array_dimensions(self) -> None:
        results = pbfe.calculate_reduced_example()
        self.assertEqual(results["lai_masses"].shape, (5, 300))
        self.assertEqual(results["tfp_masses"].shape, (300, 360))
        self.assertEqual(results["conditional_loss_exceedance"].shape, (360, 401))
        self.assertEqual(results["loss_exceedance"].shape, (401,))


if __name__ == "__main__":
    unittest.main()
