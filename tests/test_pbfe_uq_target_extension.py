"""Focused tests for the isolated PBFE uncertainty-quantification target extension."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.special import ndtr
from scipy.stats import lognorm, norm

import pbfe_uq_target_extension as extension
import pbfe_numerical_example as baseline


class UQTargetExtensionTests(unittest.TestCase):
    """Distribution, FORM, target-search, and protected-state checks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline_results = baseline.calculate_reduced_example()
        cls.form = extension.run_form_analysis()
        cls.targets = extension.run_all_target_profiles()

    def test_protected_hashes_match_preimplementation_record(self) -> None:
        check = extension.verify_protected_hashes(extension.protected_hashes())
        self.assertTrue(check["all_match"], check["mismatches"])

    def test_lognormal_arithmetic_moment_round_trip(self) -> None:
        means = np.array([0.1, 0.2, 0.5, 1.0, 1.6, 2.0])
        for cov in (0.2, 0.5):
            mu_ln, sigma_ln = baseline.lognormal_parameters_from_arithmetic_mean_cov(
                means, cov
            )
            recovered_mean = lognorm.mean(
                s=sigma_ln, loc=0.0, scale=np.exp(mu_ln)
            )
            recovered_cov = (
                lognorm.std(s=sigma_ln, loc=0.0, scale=np.exp(mu_ln))
                / recovered_mean
            )
            # 1e-12 follows the verified baseline's moment regression tolerance
            # and is orders above roundoff while still excluding parameter errors.
            np.testing.assert_allclose(recovered_mean, means, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(recovered_cov, cov, rtol=1e-12, atol=1e-12)

    def test_lai_conditional_cdf_inverse_round_trip_central_and_extreme(self) -> None:
        drivers = np.array([-40.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 40.0])
        lai = extension.conditioned_lognormal_quantile(
            drivers,
            extension.D1_LAI_ARITHMETIC_MEAN,
            baseline.LAI_ARITHMETIC_COV,
            baseline.LAI_UPPER_BOUND,
        )
        recovered = extension.conditioned_lognormal_cdf(
            lai,
            extension.D1_LAI_ARITHMETIC_MEAN,
            baseline.LAI_ARITHMETIC_COV,
            baseline.LAI_UPPER_BOUND,
        )
        # The absolute tolerance is approximately 23 binary64 epsilons and
        # covers the inverse/CDF composition at both saturated tails.
        np.testing.assert_allclose(recovered, ndtr(drivers), rtol=2e-13, atol=5e-15)

    def test_tfp_conditional_cdf_inverse_round_trip_at_several_lai_values(self) -> None:
        drivers = np.array([-12.0, -8.0, -3.0, 0.0, 3.0, 8.0, 12.0])
        for lai in (0.1, 1.6, 2.0, 4.0, 5.9):
            mean = float(baseline.tfp_arithmetic_mean(lai))
            tfp = extension.conditioned_lognormal_quantile(
                drivers,
                mean,
                baseline.TFP_ARITHMETIC_COV,
                baseline.TFP_UPPER_BOUND,
            )
            recovered = extension.conditioned_lognormal_cdf(
                tfp,
                mean,
                baseline.TFP_ARITHMETIC_COV,
                baseline.TFP_UPPER_BOUND,
            )
            np.testing.assert_allclose(recovered, ndtr(drivers), rtol=2e-13, atol=5e-15)

    def test_continuous_hierarchy_domains_without_clipping(self) -> None:
        drivers = np.array(
            [
                [-40.0, -40.0, -40.0],
                [-12.0, 12.0, -12.0],
                [-8.0, 8.0, -8.0],
                [0.0, 0.0, 0.0],
                [8.0, -8.0, 8.0],
                [12.0, 12.0, 12.0],
                [40.0, 40.0, 40.0],
            ]
        )
        values = extension.continuous_hierarchy(drivers)
        lai = np.asarray(values["lai"])
        tfp = np.asarray(values["tfp"])
        loss = np.asarray(values["dv"])
        self.assertTrue(np.all(np.isfinite(lai)))
        self.assertTrue(np.all(np.isfinite(tfp)))
        self.assertTrue(np.all(np.isfinite(loss)))
        self.assertTrue(np.all((lai > 0.0) & (lai <= baseline.LAI_UPPER_BOUND)))
        self.assertTrue(np.all((tfp > 0.0) & (tfp <= baseline.TFP_UPPER_BOUND)))
        self.assertTrue(np.all(loss > 0.0))

    def test_current_lai_bin_masses_are_recovered_from_continuous_cdf(self) -> None:
        recovered = extension.conditioned_lognormal_bin_masses(
            baseline.LAI_ARITHMETIC_MEANS,
            baseline.LAI_ARITHMETIC_COV,
            baseline.LAI_UPPER_BOUND,
            self.baseline_results["lai_edges"],
        )
        # The measured maximum analytical-vs-SciPy-CDF difference is 3.3e-16;
        # 1e-15 is a floating-resolution bound, not a loose scientific tolerance.
        np.testing.assert_allclose(
            recovered,
            self.baseline_results["lai_masses"],
            rtol=1e-12,
            atol=1e-15,
        )

    def test_all_current_tfp_bin_masses_are_recovered(self) -> None:
        recovered = extension.conditioned_lognormal_bin_masses(
            self.baseline_results["conditional_tfp_means"],
            baseline.TFP_ARITHMETIC_COV,
            baseline.TFP_UPPER_BOUND,
            self.baseline_results["tfp_edges"],
        )
        # The maximum absolute discrepancy is below 7e-16; the absolute term
        # handles bins whose reference mass rounds to zero.
        np.testing.assert_allclose(
            recovered,
            self.baseline_results["tfp_masses"],
            rtol=1e-11,
            atol=1e-15,
        )

    def test_current_conditional_dv_survival_probabilities_are_recovered(self) -> None:
        recovered = extension.conditional_loss_survival(
            self.baseline_results["tfp_centers"],
            self.baseline_results["loss_thresholds"],
        )
        np.testing.assert_array_equal(
            recovered, self.baseline_results["conditional_loss_exceedance"]
        )

    def test_frozen_direct_probability_is_reproduced_deterministically(self) -> None:
        self.assertEqual(
            extension.authoritative_direct_probability(),
            extension.DIRECT_PROBABILITY_REQUIRED,
        )
        self.assertEqual(
            extension.direct_scenario_probability(),
            extension.DIRECT_PROBABILITY_REQUIRED,
        )

    def test_form_safe_origin_and_sign_convention(self) -> None:
        self.assertGreater(self.form["g_at_origin"], 0.0)
        self.assertTrue(self.form["origin_safe"])
        self.assertGreater(self.form["beta"], 0.0)
        self.assertAlmostEqual(
            self.form["form_probability"], norm.cdf(-self.form["beta"]), places=15
        )

    def test_form_design_point_feasibility_alpha_norm_and_stationarity(self) -> None:
        self.assertLessEqual(
            abs(self.form["g_at_design_point"]), extension.LIMIT_STATE_TOLERANCE
        )
        self.assertLessEqual(
            abs(self.form["alpha_norm_sum"] - 1.0), extension.ALPHA_NORM_TOLERANCE
        )
        self.assertLessEqual(
            self.form["stationarity_residual_inf"], extension.STATIONARITY_TOLERANCE
        )
        point = np.asarray(self.form["governing_design_point"])
        alpha = np.asarray(self.form["alpha"])
        np.testing.assert_allclose(
            point,
            -self.form["beta"] * alpha,
            rtol=0.0,
            atol=extension.STATIONARITY_TOLERANCE * self.form["beta"],
        )

    def test_finite_difference_plateau_and_repeated_solve_convergence(self) -> None:
        convergence = self.form["finite_difference_convergence"]
        self.assertEqual(
            [row["step"] for row in convergence["rows"]],
            list(extension.FINITE_DIFFERENCE_STEPS),
        )
        self.assertTrue(convergence["plateau_stable"])
        self.assertTrue(convergence["reporting_step_on_stable_plateau"])
        self.assertEqual(len(self.form["cross_step_convergence"]), 3)
        for record in self.form["cross_step_convergence"]:
            self.assertLess(record["distance_from_primary_point"], 1e-6)
            self.assertLess(record["maximum_abs_alpha_difference_from_primary"], 1e-6)

    def test_multistart_candidates_are_retained_and_clustered_consistently(self) -> None:
        for solve in self.form["solves"].values():
            admissible = [
                index
                for index, attempt in enumerate(solve["attempts"])
                if attempt["admissible"]
            ]
            clustered = sorted(
                index
                for cluster in solve["clusters"]
                for index in cluster["member_attempt_indices"]
            )
            self.assertEqual(sorted(admissible), clustered)
            self.assertGreaterEqual(len(solve["clusters"]), 1)
            self.assertEqual(
                solve["clusters"][0]["cluster_id"], solve["governing_cluster_id"]
            )

    def test_every_target_profile_recovers_the_multiplier_one_baseline(self) -> None:
        for parameter, profile in self.targets["profiles"].items():
            baseline_points = [
                point for point in profile["profile_points"] if point["eta"] == 0.0
            ]
            self.assertEqual(len(baseline_points), 1, parameter)
            self.assertEqual(
                baseline_points[0]["probability"],
                extension.DIRECT_PROBABILITY_REQUIRED,
                parameter,
            )

    def test_every_reported_target_root_is_reproducible(self) -> None:
        root_count = 0
        for parameter, profile in self.targets["profiles"].items():
            for root in profile["roots"]:
                root_count += 1
                recomputed = extension.direct_scenario_probability(
                    {parameter: root["multiplier"]}
                )
                self.assertAlmostEqual(recomputed, root["probability"], places=14)
                self.assertLessEqual(
                    abs(recomputed - extension.TARGET_PROBABILITY),
                    extension.TARGET_ROOT_PROBABILITY_TOLERANCE,
                )
                self.assertTrue(root["probability_resolution_pass"])
                self.assertTrue(root["relative_multiplier_resolution_pass"])
        self.assertGreater(root_count, 0)
        self.assertFalse(self.targets["two_dimensional_contingency_triggered"])

    def test_profile_monotonicity_is_observed_and_not_assumed(self) -> None:
        for profile in self.targets["profiles"].values():
            self.assertEqual(
                profile["observed_nonmonotonic"],
                extension._observed_nonmonotonic(profile["profile_points"]),
            )


if __name__ == "__main__":
    unittest.main()
