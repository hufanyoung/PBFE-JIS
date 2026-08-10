"""Deterministic implementation of the reduced PBFE numerical illustration.

The calculation is synthetic and demonstrates probability propagation only:

    D0--D4 environmental-hazard state (IM) -> LAI (PRP) -> TFP proxy (Q) -> normalized loss (DV)

It does not implement or calibrate an agricultural damage measure or state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import lognorm


# ---------------------------------------------------------------------------
# Synthetic illustrative assumptions
# ---------------------------------------------------------------------------

DROUGHT_LABELS = ("D0", "D1", "D2", "D3", "D4")
DROUGHT_PROBABILITIES = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

# Arithmetic moments of the underlying (untruncated) LAI lognormal models;
# dimensionless LAI. Propagation then conditions each model on (0, 6].
LAI_ARITHMETIC_MEANS = np.array([1.8, 1.6, 1.4, 1.2, 1.0])
LAI_ARITHMETIC_COV = 0.5
LAI_UPPER_BOUND = 6.0
LAI_BIN_WIDTH = 0.02

# TFP is a dimensionless illustrative productivity proxy, not measured yield.
TFP_ARITHMETIC_COV = 0.2
TFP_UPPER_BOUND = 1.8
TFP_BIN_WIDTH = 0.005

# DV is a dimensionless illustrative normalized-loss quantity.
LOSS_ARITHMETIC_COV = 0.2
LOSS_MAXIMUM_DISPLAY_THRESHOLD = 2.0
LOSS_THRESHOLD_INCREMENT = 0.005

DISPLAY_LAI_VALUES = (0.1, 2.0, 4.0)
DISPLAY_TFP_VALUES = (0.5, 1.0, 1.5)
SELECTED_LOSS_THRESHOLDS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)

FIGURE_FILENAMES = {
    "lai": "illustrative_lai_distributions.png",
    "tfp": "illustrative_tfp_distributions.png",
    "loss": "illustrative_loss_distributions.png",
    "exceedance": "synthetic_loss_exceedance.png",
}


def lognormal_parameters_from_arithmetic_mean_cov(
    arithmetic_mean: np.ndarray | float,
    arithmetic_cov: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert arithmetic mean and COV to log-space Normal parameters.

    If ln(X) ~ Normal(mu_ln, sigma_ln**2), SciPy uses
    ``lognorm(s=sigma_ln, loc=0, scale=exp(mu_ln))``.
    """

    mean = np.asarray(arithmetic_mean, dtype=float)
    if np.any(mean <= 0.0):
        raise ValueError("All arithmetic means must be strictly positive.")
    if not np.isfinite(arithmetic_cov) or arithmetic_cov <= 0.0:
        raise ValueError("The arithmetic COV must be finite and positive.")

    sigma_ln_squared = np.log1p(arithmetic_cov**2)
    sigma_ln = np.full_like(mean, np.sqrt(sigma_ln_squared), dtype=float)
    mu_ln = np.log(mean) - 0.5 * sigma_ln_squared
    return mu_ln, sigma_ln


def uniform_bin_grid(upper_bound: float, bin_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return bin edges and centers on (0, upper_bound]."""

    if upper_bound <= 0.0 or bin_width <= 0.0:
        raise ValueError("Grid bounds and widths must be positive.")
    number_of_bins = int(round(upper_bound / bin_width))
    if not np.isclose(number_of_bins * bin_width, upper_bound):
        raise ValueError("The bin width must divide the upper bound exactly.")
    edges = np.linspace(0.0, upper_bound, number_of_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def normalized_lognormal_bin_masses(
    mu_ln: np.ndarray,
    sigma_ln: np.ndarray,
    bin_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized bin masses using exact CDF differences.

    The returned distribution is the lognormal distribution conditioned on
    the finite interval represented by ``bin_edges``. ``domain_probability``
    records the unconditioned mass retained by that interval.
    """

    mu = np.atleast_1d(np.asarray(mu_ln, dtype=float))
    sigma = np.atleast_1d(np.asarray(sigma_ln, dtype=float))
    edges = np.asarray(bin_edges, dtype=float)
    if mu.shape != sigma.shape:
        raise ValueError("mu_ln and sigma_ln must have the same shape.")
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must be a strictly increasing vector.")

    scales = np.exp(mu)[:, None]
    upper_cdf = lognorm.cdf(edges[1:][None, :], s=sigma[:, None], loc=0.0, scale=scales)
    lower_cdf = lognorm.cdf(edges[:-1][None, :], s=sigma[:, None], loc=0.0, scale=scales)
    masses = upper_cdf - lower_cdf
    domain_probability = masses.sum(axis=1)
    if np.any(domain_probability <= 0.0):
        raise ValueError("The selected domain contains no probability mass.")
    normalized_masses = masses / domain_probability[:, None]
    return normalized_masses, domain_probability


def conditional_truncated_pdf(
    values: np.ndarray,
    arithmetic_mean: float,
    arithmetic_cov: float,
    upper_bound: float,
) -> np.ndarray:
    """Evaluate a lognormal PDF conditioned on 0 < X <= upper_bound."""

    mu_ln, sigma_ln = lognormal_parameters_from_arithmetic_mean_cov(
        arithmetic_mean, arithmetic_cov
    )
    mu = float(mu_ln)
    sigma = float(sigma_ln)
    retained_probability = lognorm.cdf(
        upper_bound, s=sigma, loc=0.0, scale=np.exp(mu)
    )
    return lognorm.pdf(values, s=sigma, loc=0.0, scale=np.exp(mu)) / retained_probability


def tfp_arithmetic_mean(lai: np.ndarray | float) -> np.ndarray:
    """Synthetic LAI-to-TFP mean relationship."""

    return 0.2 + np.asarray(lai, dtype=float) / 6.0


def loss_arithmetic_mean(tfp: np.ndarray | float) -> np.ndarray:
    """Synthetic TFP-to-normalized-loss mean relationship."""

    return 1.0 - np.asarray(tfp, dtype=float) / 2.0


def calculate_reduced_example() -> Dict[str, np.ndarray]:
    """Assemble the corrected reduced-form conditional probability chain."""

    if not np.isclose(DROUGHT_PROBABILITIES.sum(), 1.0):
        raise AssertionError("Drought-category probabilities must sum to one.")
    if np.any(DROUGHT_PROBABILITIES < 0.0):
        raise AssertionError("Drought-category probabilities cannot be negative.")

    lai_edges, lai_centers = uniform_bin_grid(LAI_UPPER_BOUND, LAI_BIN_WIDTH)
    lai_mu_ln, lai_sigma_ln = lognormal_parameters_from_arithmetic_mean_cov(
        LAI_ARITHMETIC_MEANS, LAI_ARITHMETIC_COV
    )
    lai_masses, lai_domain_probability = normalized_lognormal_bin_masses(
        lai_mu_ln, lai_sigma_ln, lai_edges
    )

    conditional_tfp_means = tfp_arithmetic_mean(lai_centers)
    tfp_mu_ln, tfp_sigma_ln = lognormal_parameters_from_arithmetic_mean_cov(
        conditional_tfp_means, TFP_ARITHMETIC_COV
    )
    tfp_edges, tfp_centers = uniform_bin_grid(TFP_UPPER_BOUND, TFP_BIN_WIDTH)
    tfp_masses, tfp_domain_probability = normalized_lognormal_bin_masses(
        tfp_mu_ln, tfp_sigma_ln, tfp_edges
    )

    conditional_loss_means = loss_arithmetic_mean(tfp_centers)
    if np.any(conditional_loss_means <= 0.0):
        raise AssertionError(
            "The TFP domain produces a nonpositive lognormal loss mean."
        )
    loss_mu_ln, loss_sigma_ln = lognormal_parameters_from_arithmetic_mean_cov(
        conditional_loss_means, LOSS_ARITHMETIC_COV
    )
    loss_thresholds = np.arange(
        0.0,
        LOSS_MAXIMUM_DISPLAY_THRESHOLD + 0.5 * LOSS_THRESHOLD_INCREMENT,
        LOSS_THRESHOLD_INCREMENT,
    )
    conditional_loss_exceedance = lognorm.sf(
        loss_thresholds[None, :],
        s=loss_sigma_ln[:, None],
        loc=0.0,
        scale=np.exp(loss_mu_ln)[:, None],
    )

    expected_shapes = {
        "lai_masses": (len(DROUGHT_LABELS), len(lai_centers)),
        "tfp_masses": (len(lai_centers), len(tfp_centers)),
        "conditional_loss_exceedance": (len(tfp_centers), len(loss_thresholds)),
    }
    actual_arrays = {
        "lai_masses": lai_masses,
        "tfp_masses": tfp_masses,
        "conditional_loss_exceedance": conditional_loss_exceedance,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_arrays[name].shape != expected_shape:
            raise AssertionError(f"{name} has shape {actual_arrays[name].shape}, expected {expected_shape}.")

    loss_exceedance = (
        DROUGHT_PROBABILITIES
        @ lai_masses
        @ tfp_masses
        @ conditional_loss_exceedance
    )

    if not np.allclose(lai_masses.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("LAI conditional bin masses are not normalized.")
    if not np.allclose(tfp_masses.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("TFP conditional bin masses are not normalized.")
    if np.min(conditional_loss_exceedance) < -1e-12 or np.max(conditional_loss_exceedance) > 1.0 + 1e-12:
        raise AssertionError("Conditional loss exceedance is outside [0,1].")
    if np.any(np.diff(conditional_loss_exceedance, axis=1) > 1e-12):
        raise AssertionError("A conditional loss-exceedance curve is not monotone.")
    if np.min(loss_exceedance) < -1e-12 or np.max(loss_exceedance) > 1.0 + 1e-12:
        raise AssertionError("Final loss exceedance is outside [0,1].")
    if np.any(np.diff(loss_exceedance) > 1e-12):
        raise AssertionError("Final loss exceedance is not monotone.")
    if not np.isclose(loss_exceedance[0], 1.0, atol=1e-12):
        raise AssertionError("P(DV > 0) must equal one for the positive lognormal DV.")

    # Independent SciPy convention checks for the requested arithmetic moments.
    for means, cov, mu_ln, sigma_ln in (
        (LAI_ARITHMETIC_MEANS, LAI_ARITHMETIC_COV, lai_mu_ln, lai_sigma_ln),
        (conditional_tfp_means, TFP_ARITHMETIC_COV, tfp_mu_ln, tfp_sigma_ln),
        (conditional_loss_means, LOSS_ARITHMETIC_COV, loss_mu_ln, loss_sigma_ln),
    ):
        recovered_means = lognorm.mean(s=sigma_ln, loc=0.0, scale=np.exp(mu_ln))
        recovered_stds = lognorm.std(s=sigma_ln, loc=0.0, scale=np.exp(mu_ln))
        if not np.allclose(recovered_means, means, rtol=1e-12, atol=1e-12):
            raise AssertionError("SciPy parameters do not recover arithmetic means.")
        if not np.allclose(recovered_stds / recovered_means, cov, rtol=1e-12, atol=1e-12):
            raise AssertionError("SciPy parameters do not recover arithmetic COVs.")

    return {
        "lai_edges": lai_edges,
        "lai_centers": lai_centers,
        "lai_masses": lai_masses,
        "lai_mu_ln": lai_mu_ln,
        "lai_sigma_ln": lai_sigma_ln,
        "lai_domain_probability": lai_domain_probability,
        "tfp_edges": tfp_edges,
        "tfp_centers": tfp_centers,
        "tfp_masses": tfp_masses,
        "tfp_mu_ln": tfp_mu_ln,
        "tfp_sigma_ln": tfp_sigma_ln,
        "tfp_domain_probability": tfp_domain_probability,
        "conditional_tfp_means": conditional_tfp_means,
        "conditional_loss_means": conditional_loss_means,
        "loss_mu_ln": loss_mu_ln,
        "loss_sigma_ln": loss_sigma_ln,
        "loss_thresholds": loss_thresholds,
        "conditional_loss_exceedance": conditional_loss_exceedance,
        "loss_exceedance": loss_exceedance,
    }


def median_exceedance_threshold(thresholds: np.ndarray, exceedance: np.ndarray) -> float:
    """Interpolate the threshold at which the exceedance probability is 0.5."""

    return float(np.interp(0.5, exceedance[::-1], thresholds[::-1]))


def write_summary(results: Dict[str, np.ndarray], output_path: Path) -> None:
    """Write reproducibility diagnostics and selected curve values as JSON."""

    thresholds = results["loss_thresholds"]
    exceedance = results["loss_exceedance"]
    selected_exceedance = {
        f"{threshold:.2f}": float(np.interp(threshold, thresholds, exceedance))
        for threshold in SELECTED_LOSS_THRESHOLDS
    }
    summary = {
        "status": "synthetic reduced-form implementation-quality-control output",
        "chain": "IM -> PRP(LAI) -> Q proxy(TFP) -> DV(normalized loss)",
        "damage_state_implemented": False,
        "assumptions": {
            "drought_labels": list(DROUGHT_LABELS),
            "drought_probabilities": DROUGHT_PROBABILITIES.tolist(),
            "lai_arithmetic_means": LAI_ARITHMETIC_MEANS.tolist(),
            "lai_arithmetic_cov": LAI_ARITHMETIC_COV,
            "lai_conditioning_domain": [0.0, LAI_UPPER_BOUND],
            "tfp_mean_relationship": "0.2 + LAI / 6",
            "tfp_arithmetic_cov": TFP_ARITHMETIC_COV,
            "tfp_conditioning_domain": [0.0, TFP_UPPER_BOUND],
            "loss_mean_relationship": "1 - TFP / 2",
            "loss_arithmetic_cov": LOSS_ARITHMETIC_COV,
            "finite_domain_conditioning_note": (
                "Nominal means and COVs parameterize underlying untruncated "
                "lognormals; LAI and TFP propagation masses are then "
                "conditioned on their declared finite domains."
            ),
        },
        "diagnostics": {
            "lai_normalized_mass_max_abs_error": float(
                np.max(np.abs(results["lai_masses"].sum(axis=1) - 1.0))
            ),
            "tfp_normalized_mass_max_abs_error": float(
                np.max(np.abs(results["tfp_masses"].sum(axis=1) - 1.0))
            ),
            "lai_unconditioned_domain_probability": results[
                "lai_domain_probability"
            ].tolist(),
            "tfp_unconditioned_domain_probability_min": float(
                np.min(results["tfp_domain_probability"])
            ),
            "tfp_unconditioned_domain_probability_max": float(
                np.max(results["tfp_domain_probability"])
            ),
            "minimum_conditional_loss_mean": float(
                np.min(results["conditional_loss_means"])
            ),
            "maximum_conditional_loss_mean": float(
                np.max(results["conditional_loss_means"])
            ),
            "loss_exceedance_monotone": bool(
                np.all(np.diff(results["loss_exceedance"]) <= 1e-12)
            ),
            "loss_exceedance_min": float(np.min(results["loss_exceedance"])),
            "loss_exceedance_max": float(np.max(results["loss_exceedance"])),
        },
        "selected_loss_exceedance": selected_exceedance,
        "median_exceedance_threshold": median_exceedance_threshold(
            thresholds, exceedance
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def plot_figures(results: Dict[str, np.ndarray], output_directory: Path) -> None:
    """Generate the four corrected manuscript-facing numerical figures."""

    output_directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="darkgrid", font_scale=1.0)
    plt.rc("figure", dpi=150, figsize=(8, 6))
    plt.rc("font", size=12)

    lai_plot_values = np.linspace(0.001, LAI_UPPER_BOUND, 1000)
    lai_colors = plt.cm.YlOrRd(np.linspace(0.30, 0.95, len(DROUGHT_LABELS)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, mean, color in zip(
        DROUGHT_LABELS, LAI_ARITHMETIC_MEANS, lai_colors
    ):
        density = conditional_truncated_pdf(
            lai_plot_values, mean, LAI_ARITHMETIC_COV, LAI_UPPER_BOUND
        )
        ax.plot(lai_plot_values, density, linewidth=2.0, color=color, label=label)
    ax.set(xlabel="LAI", ylabel="PDF", xlim=(0.0, LAI_UPPER_BOUND))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / FIGURE_FILENAMES["lai"], dpi=150)
    plt.close(fig)

    tfp_plot_values = np.linspace(0.001, TFP_UPPER_BOUND, 1000)
    tfp_colors = plt.cm.YlGn(np.linspace(0.35, 0.90, len(DISPLAY_LAI_VALUES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for lai, color in zip(DISPLAY_LAI_VALUES, tfp_colors):
        mean = float(tfp_arithmetic_mean(lai))
        density = conditional_truncated_pdf(
            tfp_plot_values, mean, TFP_ARITHMETIC_COV, TFP_UPPER_BOUND
        )
        ax.plot(
            tfp_plot_values,
            density,
            linewidth=2.0,
            color=color,
            label=f"LAI = {lai:.2f}",
        )
    ax.set(xlabel="TFP proxy", ylabel="Conditional PDF", xlim=(0.0, TFP_UPPER_BOUND))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / FIGURE_FILENAMES["tfp"], dpi=150)
    plt.close(fig)

    loss_plot_values = np.linspace(0.001, LOSS_MAXIMUM_DISPLAY_THRESHOLD, 1000)
    loss_colors = ("#4C72B0", "#DD8452", "#777777")
    fig, ax = plt.subplots(figsize=(8, 6))
    for tfp, color in zip(DISPLAY_TFP_VALUES, loss_colors):
        mean = float(loss_arithmetic_mean(tfp))
        mu_ln, sigma_ln = lognormal_parameters_from_arithmetic_mean_cov(
            mean, LOSS_ARITHMETIC_COV
        )
        density = lognorm.pdf(
            loss_plot_values,
            s=float(sigma_ln),
            loc=0.0,
            scale=np.exp(float(mu_ln)),
        )
        ax.plot(
            loss_plot_values,
            density,
            linewidth=2.0,
            color=color,
            label=f"TFP = {tfp:.1f}",
        )
    ax.set(
        xlabel="DV (normalized loss)",
        ylabel="PDF",
        xlim=(0.0, LOSS_MAXIMUM_DISPLAY_THRESHOLD),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / FIGURE_FILENAMES["loss"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        results["loss_thresholds"],
        results["loss_exceedance"],
        linewidth=2.2,
        color=sns.color_palette("deep")[0],
    )
    ax.set(
        xlabel="DV (normalized loss)",
        ylabel="Probability of Exceedance",
        xlim=(0.0, LOSS_MAXIMUM_DISPLAY_THRESHOLD),
        ylim=(0.0, 1.01),
    )
    fig.tight_layout()
    fig.savefig(output_directory / FIGURE_FILENAMES["exceedance"], dpi=150)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    repository_directory = script_directory.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_directory / "figures",
        help="Directory for the four generated PNG figures.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=repository_directory / "outputs" / "revised_summary.json",
        help="Path for numerical diagnostics and selected outputs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    results = calculate_reduced_example()
    plot_figures(results, arguments.output_dir)
    write_summary(results, arguments.summary_path)
    print(f"Generated corrected figures in: {arguments.output_dir.resolve()}")
    print(f"Wrote numerical summary: {arguments.summary_path.resolve()}")
    print(
        "Median-exceedance threshold: "
        f"{median_exceedance_threshold(results['loss_thresholds'], results['loss_exceedance']):.6f}"
    )


if __name__ == "__main__":
    main()
