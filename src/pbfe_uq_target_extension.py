"""UQ and target-oriented numerical extension for the synthetic PBFE demonstration.

This isolated module implements the verified continuous hierarchy,
FORM/local directional-cosine diagnostic, Monte Carlo verification, and
inverse target-oriented synthetic parameter scenarios.  It imports and
preserves the verified discrete implementation; that discrete propagation
remains the authoritative probability calculation.

Nothing in this module is an empirical agricultural validation, a calibrated
performance standard, a real intervention model, or a decision recommendation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import optimize, special
from scipy.stats import lognorm, norm

import pbfe_numerical_example as baseline


# ---------------------------------------------------------------------------
# Verified controlling numerical settings
# ---------------------------------------------------------------------------

DIRECT_PROBABILITY_REQUIRED = 0.09591450211754649
LOSS_LIMIT = 1.0
MONTE_CARLO_SEED = 20260810
MONTE_CARLO_SAMPLE_SIZE = 1_000_000
MONTE_CARLO_BATCH_SIZE = 100_000

FINITE_DIFFERENCE_STEPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)
PRIMARY_FINITE_DIFFERENCE_STEP = 1e-4
REPEATED_SOLVE_STEPS = (1e-3, 1e-4, 1e-5)
RAY_RADII = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0)
CLUSTER_DISTANCE = 1e-4
LIMIT_STATE_TOLERANCE = 1e-8
ALPHA_NORM_TOLERANCE = 1e-10
STATIONARITY_TOLERANCE = 1e-5
PLATEAU_ALPHA_TOLERANCE = 1e-3
PLATEAU_GRADIENT_NORM_TOLERANCE = 1e-3

TARGET_PROBABILITY = 0.05
TARGET_REFERENCE_PROBABILITY = 0.10
TARGET_INITIAL_LOG_INCREMENT = 0.05
TARGET_UNIFORM_POINTS = 41
TARGET_MAX_EXPANSIONS = 9
TARGET_ROOT_PROBABILITY_TOLERANCE = 1e-6
TARGET_ROOT_RELATIVE_MULTIPLIER_TOLERANCE = 1e-5

D1_INDEX = baseline.DROUGHT_LABELS.index("D1")
D1_LAI_ARITHMETIC_MEAN = float(baseline.LAI_ARITHMETIC_MEANS[D1_INDEX])

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "uq_target_extension"


# Recorded before implementation. These artifacts are protected and must stay
# byte-identical. Paths are relative to PROJECT_ROOT.
EXPECTED_PROTECTED_SHA256: Mapping[str, str] = {
    "src/pbfe_numerical_example.py":
        "78c24a54471ed800e1af9a13c655d63288659b16e74655dd205c68262a1437c7",
    "tests/test_pbfe_numerical_example.py":
        "78f328b125dbae9e7c3df091b8a57bfc554db276016e29da8fcc170cd1344515",
    "outputs/numerical_summary.json":
        "49cfb949d6887cc02b8087d950c0a0285883274563eb9bbf1be9a227a3baa0ee",
    "figures/illustrative_lai_distributions.png":
        "7ed3ad9a73d0724848cb49cadb8e878ace35cc54e1d47239a158e912156ee796",
    "figures/illustrative_tfp_distributions.png":
        "16eb8d19f579700ba1149b62672d443739c45c57d97bdd064bae61a60cd0ded5",
    "figures/illustrative_loss_distributions.png":
        "98eb6560e2b43c4aca4e299f7b639343250b67fadab237fb940937a6ea00a3f4",
    "figures/synthetic_loss_exceedance.png":
        "f02a703ec369fcf8192289a8a1f16b8a74092f3db0639fd6afd8fe4ee15ec2c8",
}


TARGET_PARAMETER_METADATA: Mapping[str, Mapping[str, str]] = {
    "r_mL": {"module": "LAI", "quantity": "nominal arithmetic mean"},
    "r_cL": {"module": "LAI", "quantity": "arithmetic COV"},
    "r_mT": {"module": "TFP link", "quantity": "nominal conditional mean"},
    "r_cT": {"module": "TFP link", "quantity": "conditional arithmetic COV"},
    "r_mD": {"module": "DV/loss link", "quantity": "nominal conditional mean"},
    "r_cD": {"module": "DV/loss link", "quantity": "conditional arithmetic COV"},
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _as_builtin(value: Any) -> Any:
    """Recursively convert NumPy/scientific values into JSON-safe objects."""

    if isinstance(value, np.ndarray):
        return [_as_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _as_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes() -> Dict[str, str]:
    """Hash every protected baseline numerical artifact."""

    return {
        relative_path: sha256_file(PROJECT_ROOT / relative_path)
        for relative_path in EXPECTED_PROTECTED_SHA256
    }


def verify_protected_hashes(hashes: Mapping[str, str]) -> Dict[str, Any]:
    """Compare current protected hashes with their pre-implementation values."""

    mismatches = {
        path: {"expected": EXPECTED_PROTECTED_SHA256[path], "actual": hashes.get(path)}
        for path in EXPECTED_PROTECTED_SHA256
        if hashes.get(path) != EXPECTED_PROTECTED_SHA256[path]
    }
    return {"all_match": not mismatches, "mismatches": mismatches}


def authoritative_direct_probability() -> float:
    """Recompute the protected direct discrete probability at DV = 1."""

    results = baseline.calculate_reduced_example()
    indices = np.flatnonzero(
        np.isclose(results["loss_thresholds"], LOSS_LIMIT, rtol=0.0, atol=1e-15)
    )
    if len(indices) != 1:
        raise RuntimeError("The protected loss grid does not contain exactly one DV=1 threshold.")
    value = float(results["loss_exceedance"][indices[0]])
    if value != DIRECT_PROBABILITY_REQUIRED:
        raise RuntimeError(
            "Protected direct baseline changed: "
            f"computed {value!r}, required {DIRECT_PROBABILITY_REQUIRED!r}."
        )
    return value


# ---------------------------------------------------------------------------
# Exact conditioned-lognormal maps and continuous PBFE hierarchy
# ---------------------------------------------------------------------------


def standard_normal_quantile_from_log_cdf(log_probability: np.ndarray | float) -> np.ndarray | float:
    """Evaluate Phi^{-1}(exp(log_probability)) without arbitrary clipping.

    The upper branch uses 1-p = -expm1(log(p)); the extreme lower branch
    solves log(Phi(z)) = log(p) when exp(log(p)) would underflow.
    """

    values = np.asarray(log_probability, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values >= 0.0):
        raise ValueError("Log-CDF probabilities must be finite and strictly negative.")

    result = np.empty_like(values)
    log_half = math.log(0.5)
    log_smallest_subnormal = math.log(float(np.nextafter(0.0, 1.0)))

    upper_mask = values > log_half
    if np.any(upper_mask):
        upper_tail_probability = -np.expm1(values[upper_mask])
        if np.any(upper_tail_probability <= 0.0):
            raise FloatingPointError("Upper-tail probability lost numerical resolution.")
        result[upper_mask] = -special.ndtri(upper_tail_probability)

    ordinary_lower_mask = (values <= log_half) & (values >= log_smallest_subnormal)
    if np.any(ordinary_lower_mask):
        result[ordinary_lower_mask] = special.ndtri(np.exp(values[ordinary_lower_mask]))

    extreme_indices = np.argwhere(values < log_smallest_subnormal)
    for raw_index in extreme_indices:
        index = tuple(raw_index)
        target = float(values[index])
        lower = -8.0
        while float(special.log_ndtr(lower)) > target:
            lower *= 2.0
        result[index] = optimize.brentq(
            lambda z: float(special.log_ndtr(z)) - target,
            lower,
            0.0,
            xtol=5e-14,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=200,
        )

    if result.ndim == 0:
        return float(result)
    return result


def conditioned_lognormal_quantile(
    u: np.ndarray | float,
    arithmetic_mean: np.ndarray | float,
    arithmetic_cov: float,
    upper_bound: float,
) -> np.ndarray | float:
    """Map a standard-normal driver to Lognormal(mean,COV) | X <= b."""

    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("The conditioning upper bound must be finite and positive.")
    u_values = np.asarray(u, dtype=float)
    if np.any(~np.isfinite(u_values)):
        raise ValueError("Standard-normal drivers must be finite.")

    mu_ln, sigma_ln = baseline.lognormal_parameters_from_arithmetic_mean_cov(
        arithmetic_mean, arithmetic_cov
    )
    mu_values, sigma_values, driver_values = np.broadcast_arrays(
        np.asarray(mu_ln, dtype=float),
        np.asarray(sigma_ln, dtype=float),
        u_values,
    )
    upper_reduced = (math.log(upper_bound) - mu_values) / sigma_values
    log_conditional_probability = special.log_ndtr(upper_reduced) + special.log_ndtr(
        driver_values
    )
    reduced_quantile = np.asarray(
        standard_normal_quantile_from_log_cdf(log_conditional_probability), dtype=float
    )

    # In the far upper tail, inverse-CDF roundoff can place z a few ulps above
    # a even though log(A Phi(U)) <= log(A) mathematically. Re-solve the same
    # log-CDF equation with a as its exact bracket endpoint. This is equation
    # enforcement, not probability/sample clipping.
    upper_roundoff = reduced_quantile > upper_reduced
    for raw_index in np.argwhere(upper_roundoff):
        index = tuple(raw_index)
        target = float(log_conditional_probability[index])
        upper = float(upper_reduced[index])
        lower = min(-8.0, upper - 1.0)
        while float(special.log_ndtr(lower)) > target:
            lower *= 2.0
        reduced_quantile[index] = optimize.brentq(
            lambda z: float(special.log_ndtr(z)) - target,
            lower,
            upper,
            xtol=5e-14,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=200,
        )

    # The upper-bound-centered algebraic form preserves log(X) <= log(b)
    # without altering the requested conditional probability.
    log_values = math.log(upper_bound) + sigma_values * (
        reduced_quantile - upper_reduced
    )
    values = np.exp(log_values)

    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise FloatingPointError("Conditioned lognormal quantile is not finite and positive.")
    roundoff_allowance = 8.0 * abs(float(np.spacing(upper_bound)))
    if np.any(values > upper_bound + roundoff_allowance):
        raise FloatingPointError("Conditioned lognormal quantile exceeds its domain.")

    if values.ndim == 0:
        return float(values)
    return values


def conditioned_lognormal_cdf(
    values: np.ndarray | float,
    arithmetic_mean: np.ndarray | float,
    arithmetic_cov: float,
    upper_bound: float,
) -> np.ndarray | float:
    """Evaluate the exact CDF of Lognormal(mean,COV) conditioned on X <= b."""

    x = np.asarray(values, dtype=float)
    mu_ln, sigma_ln = baseline.lognormal_parameters_from_arithmetic_mean_cov(
        arithmetic_mean, arithmetic_cov
    )
    x_values, mu_values, sigma_values = np.broadcast_arrays(
        x, np.asarray(mu_ln, dtype=float), np.asarray(sigma_ln, dtype=float)
    )
    output = np.zeros_like(x_values)
    interior = (x_values > 0.0) & (x_values < upper_bound)
    if np.any(interior):
        reduced = (np.log(x_values[interior]) - mu_values[interior]) / sigma_values[interior]
        upper_reduced = (
            math.log(upper_bound) - mu_values[interior]
        ) / sigma_values[interior]
        output[interior] = np.exp(
            special.log_ndtr(reduced) - special.log_ndtr(upper_reduced)
        )
    output[x_values >= upper_bound] = 1.0
    if output.ndim == 0:
        return float(output)
    return output


def conditioned_lognormal_bin_masses(
    arithmetic_means: np.ndarray | Sequence[float] | float,
    arithmetic_cov: float,
    upper_bound: float,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Recover finite-domain conditional masses from the analytical CDF."""

    means = np.atleast_1d(np.asarray(arithmetic_means, dtype=float))
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("Bin edges must be a strictly increasing vector.")
    if not np.isclose(edges[0], 0.0) or not np.isclose(edges[-1], upper_bound):
        raise ValueError("Bin edges must span the complete conditioned domain.")

    mu_ln, sigma_ln = baseline.lognormal_parameters_from_arithmetic_mean_cov(
        means, arithmetic_cov
    )
    upper_reduced = (np.log(upper_bound) - mu_ln) / sigma_ln
    cdf = np.zeros((len(means), len(edges)), dtype=float)
    positive_edges = edges[1:]
    reduced = (
        np.log(positive_edges)[None, :] - mu_ln[:, None]
    ) / sigma_ln[:, None]
    cdf[:, 1:] = np.exp(
        special.log_ndtr(reduced) - special.log_ndtr(upper_reduced)[:, None]
    )
    cdf[:, -1] = 1.0
    masses = np.diff(cdf, axis=1)
    if np.min(masses) < -5e-15:
        raise FloatingPointError("Analytical conditioned bin masses became negative.")
    if not np.allclose(masses.sum(axis=1), 1.0, rtol=0.0, atol=5e-15):
        raise FloatingPointError("Analytical conditioned bin masses are not normalized.")
    return masses


def continuous_hierarchy(u: np.ndarray | Sequence[float]) -> Dict[str, np.ndarray | float]:
    """Evaluate U1 -> LAI, U2 -> TFP|LAI, U3 -> DV|TFP exactly."""

    drivers = np.asarray(u, dtype=float)
    if drivers.ndim == 0 or drivers.shape[-1] != 3:
        raise ValueError("u must have final dimension 3 for U1, U2, and U3.")
    if np.any(~np.isfinite(drivers)):
        raise ValueError("All standard-normal drivers must be finite.")

    lai = conditioned_lognormal_quantile(
        drivers[..., 0],
        D1_LAI_ARITHMETIC_MEAN,
        baseline.LAI_ARITHMETIC_COV,
        baseline.LAI_UPPER_BOUND,
    )
    tfp_mean = baseline.tfp_arithmetic_mean(lai)
    tfp = conditioned_lognormal_quantile(
        drivers[..., 1],
        tfp_mean,
        baseline.TFP_ARITHMETIC_COV,
        baseline.TFP_UPPER_BOUND,
    )
    loss_mean = baseline.loss_arithmetic_mean(tfp)
    if np.any(np.asarray(loss_mean) <= 0.0):
        raise FloatingPointError("The continuous TFP value produced nonpositive loss mean.")
    loss_mu_ln, loss_sigma_ln = baseline.lognormal_parameters_from_arithmetic_mean_cov(
        loss_mean, baseline.LOSS_ARITHMETIC_COV
    )
    loss = np.exp(
        np.asarray(loss_mu_ln, dtype=float)
        + np.asarray(loss_sigma_ln, dtype=float) * drivers[..., 2]
    )
    if np.any(~np.isfinite(loss)) or np.any(loss <= 0.0):
        raise FloatingPointError("The untruncated lognormal DV value is not positive and finite.")

    if drivers.ndim == 1:
        return {"lai": float(lai), "tfp": float(tfp), "dv": float(loss)}
    return {
        "lai": np.asarray(lai),
        "tfp": np.asarray(tfp),
        "dv": np.asarray(loss),
    }


def performance_function(u: np.ndarray | Sequence[float]) -> float:
    """Loss performance function g_L(u) = 1 - DV(u)."""

    return float(LOSS_LIMIT - float(continuous_hierarchy(u)["dv"]))


def central_difference_gradient(u: np.ndarray, step: float) -> np.ndarray:
    """Central finite-difference gradient of g_L in dimensionless U-space."""

    point = np.asarray(u, dtype=float)
    if point.shape != (3,) or step <= 0.0 or not np.isfinite(step):
        raise ValueError("A finite 3-vector and positive finite step are required.")
    gradient = np.empty(3, dtype=float)
    for index in range(3):
        direction = np.zeros(3, dtype=float)
        direction[index] = step
        gradient[index] = (
            performance_function(point + direction)
            - performance_function(point - direction)
        ) / (2.0 * step)
    return gradient


def conditional_loss_survival(
    tfp_values: np.ndarray,
    thresholds: np.ndarray | float,
    loss_mean_multiplier: float = 1.0,
    loss_cov_multiplier: float = 1.0,
) -> np.ndarray:
    """Analytical untruncated lognormal survival for DV conditional on TFP."""

    tfp = np.asarray(tfp_values, dtype=float)
    if loss_mean_multiplier <= 0.0 or loss_cov_multiplier <= 0.0:
        raise ValueError("Synthetic parameter multipliers must be positive.")
    means = loss_mean_multiplier * baseline.loss_arithmetic_mean(tfp)
    cov = loss_cov_multiplier * baseline.LOSS_ARITHMETIC_COV
    if np.any(means <= 0.0):
        raise ValueError("The scenario produces a nonpositive nominal DV mean.")
    mu_ln, sigma_ln = baseline.lognormal_parameters_from_arithmetic_mean_cov(means, cov)
    threshold_values = np.atleast_1d(np.asarray(thresholds, dtype=float))
    survival = lognorm.sf(
        threshold_values[None, :],
        s=sigma_ln[:, None],
        loc=0.0,
        scale=np.exp(mu_ln)[:, None],
    )
    if np.asarray(thresholds).ndim == 0:
        return survival[:, 0]
    return survival


# ---------------------------------------------------------------------------
# Authoritative discrete scenario calculation and grid diagnostic
# ---------------------------------------------------------------------------


def baseline_multipliers() -> Dict[str, float]:
    """Return the six Eq. (S11) multipliers at their baseline value."""

    return {name: 1.0 for name in TARGET_PARAMETER_METADATA}


def direct_scenario_probability(
    multipliers: Optional[Mapping[str, float]] = None,
    lai_bin_width: Optional[float] = None,
    tfp_bin_width: Optional[float] = None,
    return_diagnostics: bool = False,
) -> float | Tuple[float, Dict[str, Any]]:
    """Compute P_direct(DV>1) for one synthetic parameter scenario.

    The matrix propagation and midpoint evaluation are the authoritative
    discrete procedure. Optional smaller bin widths are diagnostic only.
    """

    scenario = baseline_multipliers()
    if multipliers is not None:
        unknown = set(multipliers) - set(scenario)
        if unknown:
            raise ValueError(f"Unknown scenario multipliers: {sorted(unknown)}")
        scenario.update({key: float(value) for key, value in multipliers.items()})
    if any((not np.isfinite(value) or value <= 0.0) for value in scenario.values()):
        raise ValueError("Every synthetic parameter multiplier must be finite and positive.")

    lai_width = baseline.LAI_BIN_WIDTH if lai_bin_width is None else float(lai_bin_width)
    tfp_width = baseline.TFP_BIN_WIDTH if tfp_bin_width is None else float(tfp_bin_width)
    lai_edges, lai_centers = baseline.uniform_bin_grid(
        baseline.LAI_UPPER_BOUND, lai_width
    )
    tfp_edges, tfp_centers = baseline.uniform_bin_grid(
        baseline.TFP_UPPER_BOUND, tfp_width
    )

    lai_means = np.asarray(baseline.LAI_ARITHMETIC_MEANS, dtype=float).copy()
    lai_means[D1_INDEX] *= scenario["r_mL"]
    lai_cov = baseline.LAI_ARITHMETIC_COV * scenario["r_cL"]
    lai_mu_ln, lai_sigma_ln = baseline.lognormal_parameters_from_arithmetic_mean_cov(
        lai_means, lai_cov
    )
    lai_masses, lai_retained_probability = baseline.normalized_lognormal_bin_masses(
        lai_mu_ln, lai_sigma_ln, lai_edges
    )

    tfp_means = scenario["r_mT"] * baseline.tfp_arithmetic_mean(lai_centers)
    tfp_cov = baseline.TFP_ARITHMETIC_COV * scenario["r_cT"]
    tfp_mu_ln, tfp_sigma_ln = baseline.lognormal_parameters_from_arithmetic_mean_cov(
        tfp_means, tfp_cov
    )
    tfp_masses, tfp_retained_probability = baseline.normalized_lognormal_bin_masses(
        tfp_mu_ln, tfp_sigma_ln, tfp_edges
    )

    loss_survival = conditional_loss_survival(
        tfp_centers,
        LOSS_LIMIT,
        loss_mean_multiplier=scenario["r_mD"],
        loss_cov_multiplier=scenario["r_cD"],
    )
    probability = float(
        baseline.DROUGHT_PROBABILITIES @ lai_masses @ tfp_masses @ loss_survival
    )
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise FloatingPointError("The discrete scenario probability is outside [0,1].")

    diagnostics = {
        "multipliers": scenario,
        "lai_bin_width": lai_width,
        "tfp_bin_width": tfp_width,
        "lai_bin_count": len(lai_centers),
        "tfp_bin_count": len(tfp_centers),
        "lai_mass_max_abs_normalization_error": float(
            np.max(np.abs(lai_masses.sum(axis=1) - 1.0))
        ),
        "tfp_mass_max_abs_normalization_error": float(
            np.max(np.abs(tfp_masses.sum(axis=1) - 1.0))
        ),
        "lai_retained_probability_D1": float(lai_retained_probability[D1_INDEX]),
        "tfp_retained_probability_min": float(np.min(tfp_retained_probability)),
        "tfp_retained_probability_max": float(np.max(tfp_retained_probability)),
        "probability": probability,
    }
    if return_diagnostics:
        return probability, diagnostics
    return probability


def grid_refinement_diagnostic() -> List[Dict[str, Any]]:
    """Evaluate non-destructive midpoint-grid refinements at DV = 1."""

    records: List[Dict[str, Any]] = []
    for refinement_factor in (1, 2, 4):
        probability, diagnostics = direct_scenario_probability(
            lai_bin_width=baseline.LAI_BIN_WIDTH / refinement_factor,
            tfp_bin_width=baseline.TFP_BIN_WIDTH / refinement_factor,
            return_diagnostics=True,
        )
        records.append(
            {
                "refinement_factor": refinement_factor,
                "authoritative_baseline": refinement_factor == 1,
                **diagnostics,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Monte Carlo verification
# ---------------------------------------------------------------------------


def monte_carlo_verification(
    seed: int = MONTE_CARLO_SEED,
    sample_size: int = MONTE_CARLO_SAMPLE_SIZE,
    batch_size: int = MONTE_CARLO_BATCH_SIZE,
) -> Dict[str, Any]:
    """Run deterministic independent Monte Carlo on the continuous hierarchy."""

    if sample_size <= 0 or batch_size <= 0:
        raise ValueError("Monte Carlo sample and batch sizes must be positive.")
    rng = np.random.default_rng(seed)
    exceedance_count = 0
    batch_counts: List[int] = []
    completed = 0
    while completed < sample_size:
        current_size = min(batch_size, sample_size - completed)
        drivers = rng.standard_normal((current_size, 3))
        loss = np.asarray(continuous_hierarchy(drivers)["dv"])
        current_count = int(np.count_nonzero(loss > LOSS_LIMIT))
        batch_counts.append(current_count)
        exceedance_count += current_count
        completed += current_size

    probability = exceedance_count / sample_size
    standard_error = math.sqrt(probability * (1.0 - probability) / sample_size)
    interval = (
        max(0.0, probability - 1.96 * standard_error),
        min(1.0, probability + 1.96 * standard_error),
    )
    direct = authoritative_direct_probability()
    signed_difference = probability - direct
    return {
        "seed": int(seed),
        "sample_size": int(sample_size),
        "batch_size": int(batch_size),
        "batch_exceedance_counts": batch_counts,
        "exceedance_count": int(exceedance_count),
        "probability": float(probability),
        "bernoulli_standard_error": float(standard_error),
        "normal_approximation_95_percent_interval": [float(interval[0]), float(interval[1])],
        "signed_difference_mc_minus_direct": float(signed_difference),
        "absolute_difference_from_direct": float(abs(signed_difference)),
        "relative_difference_from_direct": float(signed_difference / direct),
        "direct_outside_auxiliary_95_percent_interval": bool(
            direct < interval[0] or direct > interval[1]
        ),
    }


# ---------------------------------------------------------------------------
# FORM, directional cosines, and multiple-start diagnostics
# ---------------------------------------------------------------------------


def finite_difference_convergence(point: np.ndarray) -> Dict[str, Any]:
    """Evaluate all configured gradient steps and identify stable 3-step windows."""

    rows: List[Dict[str, Any]] = []
    for step in FINITE_DIFFERENCE_STEPS:
        gradient = central_difference_gradient(point, step)
        norm_value = float(np.linalg.norm(gradient))
        if not np.isfinite(norm_value) or norm_value <= 0.0:
            alpha = np.full(3, np.nan)
        else:
            alpha = gradient / norm_value
        rows.append(
            {
                "step": float(step),
                "gradient": gradient.tolist(),
                "gradient_norm": norm_value,
                "alpha": alpha.tolist(),
            }
        )

    stable_windows: List[Dict[str, Any]] = []
    primary_index = FINITE_DIFFERENCE_STEPS.index(PRIMARY_FINITE_DIFFERENCE_STEP)
    for start_index in range(len(rows) - 2):
        window = rows[start_index : start_index + 3]
        alphas = np.asarray([row["alpha"] for row in window], dtype=float)
        norms = np.asarray([row["gradient_norm"] for row in window], dtype=float)
        if np.any(~np.isfinite(alphas)) or np.any(~np.isfinite(norms)):
            continue
        alpha_change = float(np.max(np.abs(np.diff(alphas, axis=0))))
        norm_denominator = max(float(np.max(np.abs(norms))), np.finfo(float).tiny)
        relative_norm_change = float(np.max(np.abs(np.diff(norms))) / norm_denominator)
        stable = (
            alpha_change <= PLATEAU_ALPHA_TOLERANCE
            and relative_norm_change <= PLATEAU_GRADIENT_NORM_TOLERANCE
        )
        if stable:
            indices = list(range(start_index, start_index + 3))
            stable_windows.append(
                {
                    "steps": [rows[index]["step"] for index in indices],
                    "maximum_adjacent_componentwise_alpha_change": alpha_change,
                    "maximum_adjacent_relative_gradient_norm_change": relative_norm_change,
                    "contains_reporting_step_1e-4": primary_index in indices,
                }
            )

    return {
        "rows": rows,
        "stable_windows": stable_windows,
        "plateau_stable": bool(stable_windows),
        "reporting_step_on_stable_plateau": any(
            window["contains_reporting_step_1e-4"] for window in stable_windows
        ),
        "alpha_tolerance": PLATEAU_ALPHA_TOLERANCE,
        "relative_gradient_norm_tolerance": PLATEAU_GRADIENT_NORM_TOLERANCE,
    }


def _ray_directions() -> Iterable[Tuple[int, int, int]]:
    for direction in itertools.product((-1, 0, 1), repeat=3):
        if direction != (0, 0, 0):
            yield direction


def generate_form_starts(
    gradient_step: float,
    radii: Sequence[float] = RAY_RADII,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate origin, projection, and every feasible configured ray-root start."""

    origin = np.zeros(3, dtype=float)
    starts: List[Dict[str, Any]] = [{"name": "origin", "point": origin}]
    origin_gradient = central_difference_gradient(origin, gradient_step)
    denominator = float(np.dot(origin_gradient, origin_gradient))
    if np.isfinite(denominator) and denominator > np.finfo(float).tiny:
        projection = -performance_function(origin) * origin_gradient / denominator
        starts.append({"name": "first_order_projection", "point": projection})

    ray_records: List[Dict[str, Any]] = []
    for integer_direction in _ray_directions():
        direction = np.asarray(integer_direction, dtype=float)
        direction /= np.linalg.norm(direction)
        values = [performance_function(radius * direction) for radius in radii]
        roots: List[float] = []
        for left, right, f_left, f_right in zip(
            radii[:-1], radii[1:], values[:-1], values[1:]
        ):
            if f_left == 0.0:
                roots.append(float(left))
            elif f_left * f_right < 0.0:
                roots.append(
                    float(
                        optimize.brentq(
                            lambda radius: performance_function(radius * direction),
                            left,
                            right,
                            xtol=1e-13,
                            rtol=1e-13,
                            maxiter=200,
                        )
                    )
                )
        if values[-1] == 0.0:
            roots.append(float(radii[-1]))
        unique_roots: List[float] = []
        for root in sorted(roots):
            if not unique_roots or abs(root - unique_roots[-1]) > 1e-9:
                unique_roots.append(root)
        for root_number, root in enumerate(unique_roots, start=1):
            starts.append(
                {
                    "name": (
                        f"ray_{integer_direction[0]:+d}_{integer_direction[1]:+d}_"
                        f"{integer_direction[2]:+d}_root_{root_number}"
                    ),
                    "point": root * direction,
                }
            )
        ray_records.append(
            {
                "integer_direction": list(integer_direction),
                "unit_direction": direction.tolist(),
                "radii": [float(value) for value in radii],
                "g_values": [float(value) for value in values],
                "root_radii": unique_roots,
            }
        )
    return starts, ray_records


def _evaluate_solver_candidate(
    result: optimize.OptimizeResult,
    start: Mapping[str, Any],
    solver_method: str,
    solver_gradient_step: float,
) -> Dict[str, Any]:
    point = np.asarray(result.x, dtype=float)
    finite = bool(np.all(np.isfinite(point)))
    beta = float(np.linalg.norm(point)) if finite else float("nan")
    residual = performance_function(point) if finite else float("nan")
    convergence = finite_difference_convergence(point) if finite else {
        "rows": [],
        "stable_windows": [],
        "plateau_stable": False,
        "reporting_step_on_stable_plateau": False,
    }
    gradient = (
        central_difference_gradient(point, PRIMARY_FINITE_DIFFERENCE_STEP)
        if finite
        else np.full(3, np.nan)
    )
    gradient_norm = float(np.linalg.norm(gradient))
    gradient_defined = bool(np.isfinite(gradient_norm) and gradient_norm > 0.0)
    alpha = gradient / gradient_norm if gradient_defined else np.full(3, np.nan)
    alpha_norm_sum = float(np.dot(alpha, alpha)) if gradient_defined else float("nan")
    stationarity_residual = (
        float(np.linalg.norm(point / beta + alpha, ord=np.inf))
        if gradient_defined and np.isfinite(beta) and beta > 0.0
        else float("nan")
    )

    checks = {
        "solver_success": bool(result.success),
        "finite_values": finite and np.isfinite(residual),
        "limit_state_residual": bool(
            np.isfinite(residual)
            and abs(residual) <= LIMIT_STATE_TOLERANCE * max(1.0, LOSS_LIMIT)
        ),
        "normalized_gradient_defined": gradient_defined,
        "alpha_norm": bool(
            np.isfinite(alpha_norm_sum)
            and abs(alpha_norm_sum - 1.0) <= ALPHA_NORM_TOLERANCE
        ),
        "stationarity": bool(
            np.isfinite(stationarity_residual)
            and stationarity_residual <= STATIONARITY_TOLERANCE
        ),
        "finite_difference_plateau": bool(convergence["plateau_stable"]),
        "reporting_step_on_stable_plateau": bool(
            convergence["reporting_step_on_stable_plateau"]
        ),
    }
    admissible = all(checks.values())
    return {
        "start_name": str(start["name"]),
        "start_point": np.asarray(start["point"], dtype=float).tolist(),
        "solver_method": solver_method,
        "solver_gradient_step": float(solver_gradient_step),
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "iterations": int(getattr(result, "nit", -1)),
        "function_evaluations": int(getattr(result, "nfev", -1)),
        "point": point.tolist(),
        "objective": float(0.5 * np.dot(point, point)) if finite else float("nan"),
        "beta": beta,
        "g": float(residual),
        "reporting_gradient_step": PRIMARY_FINITE_DIFFERENCE_STEP,
        "gradient": gradient.tolist(),
        "gradient_norm": gradient_norm,
        "alpha": alpha.tolist(),
        "absolute_alpha": np.abs(alpha).tolist(),
        "alpha_norm_sum": alpha_norm_sum,
        "stationarity_residual_inf": stationarity_residual,
        "finite_difference_convergence": convergence,
        "checks": checks,
        "admissible": admissible,
    }


def _cluster_admissible_candidates(attempts: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    admissible_indices = [
        index for index, attempt in enumerate(attempts) if bool(attempt["admissible"])
    ]
    admissible_indices.sort(key=lambda index: float(attempts[index]["beta"]))
    clusters: List[Dict[str, Any]] = []
    for attempt_index in admissible_indices:
        point = np.asarray(attempts[attempt_index]["point"], dtype=float)
        assigned = False
        for cluster in clusters:
            representative_point = np.asarray(cluster["representative_point"], dtype=float)
            if float(np.linalg.norm(point - representative_point)) <= CLUSTER_DISTANCE:
                cluster["member_attempt_indices"].append(attempt_index)
                cluster["found_by"].append(attempts[attempt_index]["start_name"])
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "representative_attempt_index": attempt_index,
                    "representative_point": point.tolist(),
                    "beta": float(attempts[attempt_index]["beta"]),
                    "g": float(attempts[attempt_index]["g"]),
                    "member_attempt_indices": [attempt_index],
                    "found_by": [attempts[attempt_index]["start_name"]],
                }
            )
    for cluster_number, cluster in enumerate(clusters, start=1):
        cluster["cluster_id"] = cluster_number
        cluster["member_count"] = len(cluster["member_attempt_indices"])
    return clusters


def solve_design_point(gradient_step: float) -> Dict[str, Any]:
    """Solve the configured multi-start constrained FORM problem."""

    starts, ray_records = generate_form_starts(gradient_step)
    attempts: List[Dict[str, Any]] = []
    constraint = {
        "type": "eq",
        "fun": performance_function,
        "jac": lambda point: central_difference_gradient(point, gradient_step),
    }
    for start in starts:
        result = optimize.minimize(
            fun=lambda point: 0.5 * float(np.dot(point, point)),
            x0=np.asarray(start["point"], dtype=float),
            jac=lambda point: np.asarray(point, dtype=float),
            constraints=(constraint,),
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )
        attempts.append(
            _evaluate_solver_candidate(result, start, "SLSQP", gradient_step)
        )

    clusters = _cluster_admissible_candidates(attempts)
    fallback_triggered = not clusters or min(cluster["beta"] for cluster in clusters) >= 11.5
    fallback_reason: Optional[str] = None
    if fallback_triggered:
        fallback_reason = "no admissible SLSQP candidate" if not clusters else "candidate near ray envelope"
        fallback_start = (
            np.asarray(starts[1]["point"], dtype=float)
            if len(starts) > 1
            else np.zeros(3, dtype=float)
        )
        nonlinear_constraint = optimize.NonlinearConstraint(
            performance_function,
            0.0,
            0.0,
            jac=lambda point: central_difference_gradient(point, gradient_step),
        )
        result = optimize.minimize(
            fun=lambda point: 0.5 * float(np.dot(point, point)),
            x0=fallback_start,
            jac=lambda point: np.asarray(point, dtype=float),
            constraints=(nonlinear_constraint,),
            method="trust-constr",
            options={
                "maxiter": 1000,
                "gtol": 1e-10,
                "xtol": 1e-12,
                "barrier_tol": 1e-12,
                "verbose": 0,
            },
        )
        attempts.append(
            _evaluate_solver_candidate(
                result,
                {"name": "trust_constr_fallback", "point": fallback_start},
                "trust-constr",
                gradient_step,
            )
        )
        clusters = _cluster_admissible_candidates(attempts)

    if not clusters:
        raise RuntimeError(
            f"No admissible design-point candidate at finite-difference step {gradient_step:g}."
        )
    clusters.sort(key=lambda cluster: float(cluster["beta"]))
    governing_cluster = clusters[0]
    governing_attempt = attempts[int(governing_cluster["representative_attempt_index"])]
    return {
        "solver_gradient_step": float(gradient_step),
        "start_count": len(starts),
        "starts": [
            {"name": start["name"], "point": np.asarray(start["point"]).tolist()}
            for start in starts
        ],
        "ray_search": ray_records,
        "attempts": attempts,
        "clusters": clusters,
        "governing_cluster_id": int(governing_cluster["cluster_id"]),
        "governing_attempt": governing_attempt,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
    }


def run_form_analysis() -> Dict[str, Any]:
    """Run primary and repeated-step FORM solves and assemble diagnostics."""

    origin = np.zeros(3, dtype=float)
    g_origin = performance_function(origin)
    if not g_origin > 0.0:
        raise RuntimeError(
            "Safe-origin convention failed: g_L([0,0,0]) is not positive."
        )

    solves: Dict[str, Dict[str, Any]] = {}
    for step in REPEATED_SOLVE_STEPS:
        solves[f"{step:.0e}"] = solve_design_point(step)
    primary = solves[f"{PRIMARY_FINITE_DIFFERENCE_STEP:.0e}"]["governing_attempt"]
    beta = float(primary["beta"])
    form_probability = float(norm.cdf(-beta))

    cross_step: List[Dict[str, Any]] = []
    primary_point = np.asarray(primary["point"], dtype=float)
    primary_alpha = np.asarray(primary["alpha"], dtype=float)
    for key, solve in solves.items():
        candidate = solve["governing_attempt"]
        cross_step.append(
            {
                "solver_gradient_step": float(solve["solver_gradient_step"]),
                "point": candidate["point"],
                "beta": float(candidate["beta"]),
                "alpha_at_reporting_step": candidate["alpha"],
                "distance_from_primary_point": float(
                    np.linalg.norm(np.asarray(candidate["point"]) - primary_point)
                ),
                "maximum_abs_alpha_difference_from_primary": float(
                    np.max(np.abs(np.asarray(candidate["alpha"]) - primary_alpha))
                ),
                "g": float(candidate["g"]),
                "cluster_count": len(solve["clusters"]),
            }
        )

    return {
        "convention": {
            "performance_function": "g_L(u) = 1.0 - DV(u)",
            "undesirable_region": "g_L < 0, equivalent to DV > 1.0",
            "beta": "positive Euclidean distance ||u*|| for verified g_L(0)>0",
            "form_probability": "Phi(-beta)",
            "alpha": "grad_u(g_L) / ||grad_u(g_L)|| at u*; local to DV=1",
        },
        "g_at_origin": float(g_origin),
        "origin_safe": True,
        "primary_solver_gradient_step": PRIMARY_FINITE_DIFFERENCE_STEP,
        "beta": beta,
        "form_probability": form_probability,
        "governing_design_point": primary["point"],
        "g_at_design_point": float(primary["g"]),
        "gradient": primary["gradient"],
        "alpha": primary["alpha"],
        "absolute_alpha": primary["absolute_alpha"],
        "alpha_norm_sum": float(primary["alpha_norm_sum"]),
        "stationarity_residual_inf": float(primary["stationarity_residual_inf"]),
        "finite_difference_convergence": primary["finite_difference_convergence"],
        "cross_step_convergence": cross_step,
        "solves": solves,
    }


# ---------------------------------------------------------------------------
# Target-oriented one-dimensional scenario searches
# ---------------------------------------------------------------------------


def _observed_nonmonotonic(profile_points: Sequence[Mapping[str, Any]]) -> bool:
    finite_points = sorted(
        (
            (float(point["eta"]), float(point["probability"]))
            for point in profile_points
            if point.get("probability") is not None
        ),
        key=lambda item: item[0],
    )
    if len(finite_points) < 3:
        return False
    eta = np.asarray([point[0] for point in finite_points])
    probability = np.asarray([point[1] for point in finite_points])
    slopes = np.diff(probability) / np.diff(eta)
    material = slopes[np.abs(slopes) > 1e-10]
    return bool(np.any(material > 0.0) and np.any(material < 0.0))


def _crossing_intervals(profile_points: Sequence[Mapping[str, Any]]) -> List[Tuple[float, float]]:
    finite_points = sorted(
        (
            (float(point["eta"]), float(point["probability"]))
            for point in profile_points
            if point.get("probability") is not None
        ),
        key=lambda item: item[0],
    )
    intervals: List[Tuple[float, float]] = []
    for (eta_left, probability_left), (eta_right, probability_right) in zip(
        finite_points[:-1], finite_points[1:]
    ):
        left_value = probability_left - TARGET_PROBABILITY
        right_value = probability_right - TARGET_PROBABILITY
        if left_value == 0.0:
            intervals.append((eta_left, eta_left))
        elif left_value * right_value < 0.0:
            intervals.append((eta_left, eta_right))
    if finite_points and finite_points[-1][1] == TARGET_PROBABILITY:
        intervals.append((finite_points[-1][0], finite_points[-1][0]))
    return intervals


def target_profile(parameter: str) -> Dict[str, Any]:
    """Search one Eq. (S11) scenario family for all observed p=0.05 roots."""

    if parameter not in TARGET_PARAMETER_METADATA:
        raise ValueError(f"Unknown target parameter {parameter!r}.")

    cache: MutableMapping[float, Dict[str, Any]] = {}
    expansion_nodes: set[float] = {0.0}
    adaptive_nodes: set[float] = set()
    issues: List[str] = []

    def evaluate_eta(eta: float) -> Optional[float]:
        key = float(eta)
        if key in cache:
            return cache[key]["probability"]
        multiplier = math.exp(key)
        try:
            probability = float(direct_scenario_probability({parameter: multiplier}))
            cache[key] = {
                "eta": key,
                "multiplier": multiplier,
                "probability": probability,
                "status": "ok",
            }
            return probability
        except (ValueError, FloatingPointError, OverflowError) as error:
            cache[key] = {
                "eta": key,
                "multiplier": multiplier,
                "probability": None,
                "status": "numerical_conditioning_issue",
                "message": str(error),
            }
            issues.append(f"eta={key:.12g}: {error}")
            return None

    baseline_probability = evaluate_eta(0.0)
    if baseline_probability is None or baseline_probability != DIRECT_PROBABILITY_REQUIRED:
        raise RuntimeError(f"Target baseline failed for {parameter}.")

    final_radius = TARGET_INITIAL_LOG_INCREMENT
    crossing_seen = False
    expansion_count = 0
    for expansion_index in range(TARGET_MAX_EXPANSIONS):
        expansion_count = expansion_index + 1
        final_radius = TARGET_INITIAL_LOG_INCREMENT * (2.0**expansion_index)
        expansion_nodes.update((-final_radius, final_radius))
        for eta in np.linspace(-final_radius, final_radius, TARGET_UNIFORM_POINTS):
            evaluate_eta(float(eta))
        points = list(cache.values())

        # Investigate every observed slope-sign change with two adaptive levels.
        for _ in range(2):
            finite = sorted(
                (
                    (float(point["eta"]), float(point["probability"]))
                    for point in cache.values()
                    if point["probability"] is not None
                ),
                key=lambda item: item[0],
            )
            if len(finite) < 3:
                break
            eta_values = np.asarray([item[0] for item in finite])
            probability_values = np.asarray([item[1] for item in finite])
            slopes = np.diff(probability_values) / np.diff(eta_values)
            new_nodes: List[float] = []
            for slope_index in range(len(slopes) - 1):
                if (
                    abs(slopes[slope_index]) > 1e-10
                    and abs(slopes[slope_index + 1]) > 1e-10
                    and slopes[slope_index] * slopes[slope_index + 1] < 0.0
                ):
                    new_nodes.extend(
                        (
                            0.5 * (eta_values[slope_index] + eta_values[slope_index + 1]),
                            0.5 * (eta_values[slope_index + 1] + eta_values[slope_index + 2]),
                        )
                    )
            if not new_nodes:
                break
            for eta in new_nodes:
                adaptive_nodes.add(float(eta))
                evaluate_eta(float(eta))

        if _crossing_intervals(list(cache.values())):
            crossing_seen = True
            break

    profile_points = sorted(cache.values(), key=lambda point: float(point["eta"]))
    intervals = _crossing_intervals(profile_points)
    roots: List[Dict[str, Any]] = []

    def target_residual(eta: float) -> float:
        probability = evaluate_eta(float(eta))
        if probability is None:
            raise FloatingPointError("Target root entered a numerically unresolved point.")
        return probability - TARGET_PROBABILITY

    for eta_left, eta_right in intervals:
        if eta_left == eta_right:
            eta_root = eta_left
        else:
            eta_root = float(
                optimize.brentq(
                    target_residual,
                    eta_left,
                    eta_right,
                    xtol=1e-12,
                    rtol=1e-12,
                    maxiter=200,
                )
            )
        probability = float(evaluate_eta(eta_root))
        multiplier = math.exp(eta_root)
        if roots and abs(eta_root - roots[-1]["eta"]) <= 1e-8:
            continue
        roots.append(
            {
                "eta": eta_root,
                "multiplier": multiplier,
                "probability": probability,
                "probability_error": probability - TARGET_PROBABILITY,
                "direction": "increase" if multiplier > 1.0 else "decrease",
                "probability_resolution_pass": abs(probability - TARGET_PROBABILITY)
                <= TARGET_ROOT_PROBABILITY_TOLERANCE,
                "relative_multiplier_resolution_bound": 1e-12,
                "relative_multiplier_resolution_pass": 1e-12
                <= TARGET_ROOT_RELATIVE_MULTIPLIER_TOLERANCE,
            }
        )

    roots.sort(key=lambda root: float(root["eta"]))
    closest_root_index: Optional[int] = None
    if roots:
        closest_root_index = int(np.argmin([abs(root["eta"]) for root in roots]))
        for index, root in enumerate(roots):
            root["closest_proportional_crossing"] = index == closest_root_index

    for point in profile_points:
        eta = float(point["eta"])
        point["is_expansion_node"] = eta in expansion_nodes
        point["is_adaptive_node"] = eta in adaptive_nodes

    nonmonotonic = _observed_nonmonotonic(profile_points)
    search_guard_reached = not crossing_seen and expansion_count == TARGET_MAX_EXPANSIONS
    status = "crossing_resolved" if roots else "no_resolved_crossing_in_evaluated_domain"
    if search_guard_reached and not issues:
        issues.append(
            "Maximum documented search expansion reached without a crossing; "
            "no claim of impossibility is made."
        )

    return {
        "parameter": parameter,
        **TARGET_PARAMETER_METADATA[parameter],
        "status": status,
        "baseline_probability": float(baseline_probability),
        "target_probability": TARGET_PROBABILITY,
        "expansion_count": expansion_count,
        "final_abs_eta": float(final_radius),
        "evaluated_eta_range": [
            float(min(point["eta"] for point in profile_points)),
            float(max(point["eta"] for point in profile_points)),
        ],
        "evaluated_multiplier_range": [
            float(min(point["multiplier"] for point in profile_points)),
            float(max(point["multiplier"] for point in profile_points)),
        ],
        "profile_points": profile_points,
        "roots": roots,
        "closest_root_index": closest_root_index,
        "crossing_exists": bool(roots),
        "observed_nonmonotonic": nonmonotonic,
        "search_guard_reached": search_guard_reached,
        "numerical_issues": sorted(set(issues)),
    }


def run_all_target_profiles() -> Dict[str, Any]:
    profiles = {
        parameter: target_profile(parameter) for parameter in TARGET_PARAMETER_METADATA
    }
    contingency_triggered = not any(
        profile["crossing_exists"] for profile in profiles.values()
    )
    contingency: Optional[Dict[str, Any]] = None
    if contingency_triggered:
        contingency = two_dimensional_contingency(profiles)
    return {
        "criteria": {
            "loss_limit": LOSS_LIMIT,
            "reference_probability": TARGET_REFERENCE_PROBABILITY,
            "target_probability": TARGET_PROBABILITY,
            "interpretation": "illustrative synthetic decision thresholds, not standards",
        },
        "search_settings": {
            "log_multiplier_initial_increment": TARGET_INITIAL_LOG_INCREMENT,
            "minimum_uniform_points": TARGET_UNIFORM_POINTS,
            "maximum_expansions": TARGET_MAX_EXPANSIONS,
            "root_probability_tolerance": TARGET_ROOT_PROBABILITY_TOLERANCE,
            "root_relative_multiplier_tolerance": TARGET_ROOT_RELATIVE_MULTIPLIER_TOLERANCE,
        },
        "profiles": profiles,
        "two_dimensional_contingency_triggered": contingency_triggered,
        "two_dimensional_contingency": contingency,
    }


def two_dimensional_contingency(one_dimensional_profiles: Mapping[str, Any]) -> Dict[str, Any]:
    """Approved contingency, called only when all six 1-D searches fail."""

    def closest_eta(profile: Mapping[str, Any]) -> float:
        finite = [
            point
            for point in profile["profile_points"]
            if point["probability"] is not None
        ]
        return float(
            min(finite, key=lambda point: abs(point["probability"] - TARGET_PROBABILITY))["eta"]
        )

    center_lai = closest_eta(one_dimensional_profiles["r_mL"])
    center_loss = closest_eta(one_dimensional_profiles["r_mD"])
    half_width_lai = max(0.1, abs(center_lai) * 0.25)
    half_width_loss = max(0.1, abs(center_loss) * 0.25)
    eta_lai = np.linspace(center_lai - half_width_lai, center_lai + half_width_lai, 41)
    eta_loss = np.linspace(center_loss - half_width_loss, center_loss + half_width_loss, 41)
    probability = np.empty((len(eta_loss), len(eta_lai)), dtype=float)
    for row, eta_d in enumerate(eta_loss):
        for column, eta_l in enumerate(eta_lai):
            probability[row, column] = direct_scenario_probability(
                {"r_mL": math.exp(float(eta_l)), "r_mD": math.exp(float(eta_d))}
            )
    return {
        "parameters": ["r_mL", "r_mD"],
        "eta_r_mL": eta_lai.tolist(),
        "eta_r_mD": eta_loss.tolist(),
        "multiplier_r_mL": np.exp(eta_lai).tolist(),
        "multiplier_r_mD": np.exp(eta_loss).tolist(),
        "probability": probability.tolist(),
        "target_boundary_present": bool(
            np.min(probability) <= TARGET_PROBABILITY <= np.max(probability)
        ),
    }


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def probability_comparison(
    direct_probability: float,
    monte_carlo: Mapping[str, Any],
    form: Mapping[str, Any],
) -> Dict[str, Any]:
    mc_probability = float(monte_carlo["probability"])
    form_probability = float(form["form_probability"])

    def comparison(first_name: str, first: float, second_name: str, second: float) -> Dict[str, Any]:
        difference = first - second
        return {
            "first": first_name,
            "second": second_name,
            "signed_first_minus_second": float(difference),
            "absolute_difference": float(abs(difference)),
            "relative_difference_using_second": float(difference / second),
        }

    return {
        "direct_is_authoritative": True,
        "direct": direct_probability,
        "continuous_monte_carlo": mc_probability,
        "form": form_probability,
        "mc_minus_direct": comparison(
            "continuous_monte_carlo", mc_probability, "direct", direct_probability
        ),
        "form_minus_direct": comparison("form", form_probability, "direct", direct_probability),
        "form_minus_mc": comparison(
            "form", form_probability, "continuous_monte_carlo", mc_probability
        ),
        "interpretation": (
            "FORM is an approximation and alpha is a local DV=1 boundary diagnostic; "
            "neither replaces the authoritative discrete probability."
        ),
    }


def write_sensitivity_table(
    path: Path,
    direct_probability: float,
    monte_carlo: Mapping[str, Any],
    form: Mapping[str, Any],
) -> None:
    driver_labels = (
        "LAI driver",
        "TFP-given-LAI driver",
        "DV-given-TFP driver",
    )
    fieldnames = [
        "driver",
        "interpretation",
        "u_star",
        "alpha",
        "absolute_alpha",
        "direct_probability_authoritative",
        "continuous_monte_carlo_probability",
        "monte_carlo_standard_error",
        "form_probability_approximate",
        "beta",
        "g_at_design_point",
        "alpha_norm_sum",
        "stationarity_residual_inf",
        "finite_difference_plateau_stable",
        "distinct_design_point_clusters",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, label in enumerate(driver_labels):
            writer.writerow(
                {
                    "driver": f"U{index + 1}",
                    "interpretation": f"local design-point importance at DV=1; {label}",
                    "u_star": form["governing_design_point"][index],
                    "alpha": form["alpha"][index],
                    "absolute_alpha": form["absolute_alpha"][index],
                    "direct_probability_authoritative": direct_probability,
                    "continuous_monte_carlo_probability": monte_carlo["probability"],
                    "monte_carlo_standard_error": monte_carlo["bernoulli_standard_error"],
                    "form_probability_approximate": form["form_probability"],
                    "beta": form["beta"],
                    "g_at_design_point": form["g_at_design_point"],
                    "alpha_norm_sum": form["alpha_norm_sum"],
                    "stationarity_residual_inf": form["stationarity_residual_inf"],
                    "finite_difference_plateau_stable": form[
                        "finite_difference_convergence"
                    ]["plateau_stable"],
                    "distinct_design_point_clusters": len(
                        form["solves"][f"{PRIMARY_FINITE_DIFFERENCE_STEP:.0e}"]["clusters"]
                    ),
                }
            )


def write_target_csv(path: Path, targets: Mapping[str, Any]) -> None:
    fieldnames = [
        "row_type",
        "parameter",
        "module",
        "quantity",
        "eta",
        "multiplier",
        "direct_probability",
        "row_status",
        "profile_status",
        "is_expansion_node",
        "is_adaptive_node",
        "direction",
        "closest_proportional_crossing",
        "probability_error",
        "probability_resolution_pass",
        "relative_multiplier_resolution_pass",
        "crossing_exists",
        "observed_nonmonotonic",
        "search_guard_reached",
        "numerical_issues",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for parameter, profile in targets["profiles"].items():
            for point in profile["profile_points"]:
                writer.writerow(
                    {
                        "row_type": "profile",
                        "parameter": parameter,
                        "module": profile["module"],
                        "quantity": profile["quantity"],
                        "eta": point["eta"],
                        "multiplier": point["multiplier"],
                        "direct_probability": point["probability"],
                        "row_status": point["status"],
                        "profile_status": profile["status"],
                        "is_expansion_node": point.get("is_expansion_node", False),
                        "is_adaptive_node": point.get("is_adaptive_node", False),
                        "direction": "",
                        "closest_proportional_crossing": "",
                        "probability_error": "",
                        "probability_resolution_pass": "",
                        "relative_multiplier_resolution_pass": "",
                        "crossing_exists": profile["crossing_exists"],
                        "observed_nonmonotonic": profile["observed_nonmonotonic"],
                        "search_guard_reached": profile["search_guard_reached"],
                        "numerical_issues": " | ".join(profile["numerical_issues"]),
                    }
                )
            for root in profile["roots"]:
                writer.writerow(
                    {
                        "row_type": "root",
                        "parameter": parameter,
                        "module": profile["module"],
                        "quantity": profile["quantity"],
                        "eta": root["eta"],
                        "multiplier": root["multiplier"],
                        "direct_probability": root["probability"],
                        "row_status": "resolved_target_crossing",
                        "profile_status": profile["status"],
                        "is_expansion_node": "",
                        "is_adaptive_node": "",
                        "direction": root["direction"],
                        "closest_proportional_crossing": root[
                            "closest_proportional_crossing"
                        ],
                        "probability_error": root["probability_error"],
                        "probability_resolution_pass": root[
                            "probability_resolution_pass"
                        ],
                        "relative_multiplier_resolution_pass": root[
                            "relative_multiplier_resolution_pass"
                        ],
                        "crossing_exists": True,
                        "observed_nonmonotonic": profile["observed_nonmonotonic"],
                        "search_guard_reached": profile["search_guard_reached"],
                        "numerical_issues": " | ".join(profile["numerical_issues"]),
                    }
                )


def plot_target_profiles(path: Path, targets: Mapping[str, Any]) -> None:
    """Write the three-panel target-oriented diagnostic figure."""

    if targets["two_dimensional_contingency_triggered"]:
        contingency = targets["two_dimensional_contingency"]
        x = np.asarray(contingency["multiplier_r_mL"])
        y = np.asarray(contingency["multiplier_r_mD"])
        probability = np.asarray(contingency["probability"])
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        contour = ax.contourf(x, y, probability, levels=16, cmap="viridis")
        if contingency["target_boundary_present"]:
            ax.contour(x, y, probability, levels=[TARGET_PROBABILITY], colors="red")
        ax.set(xscale="log", yscale="log", xlabel=r"$r_{mL}$", ylabel=r"$r_{mD}$")
        fig.colorbar(contour, ax=ax, label=r"Direct $P(DV>1)$")
        ax.set_title("Synthetic two-parameter target contingency")
    else:
        groups = (
            ("LAI", ("r_mL", "r_cL")),
            ("TFP link", ("r_mT", "r_cT")),
            ("DV/loss link", ("r_mD", "r_cD")),
        )
        colors = {"mean": "#1f77b4", "cov": "#d95f02"}
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharey=True)
        for ax, (panel_title, parameters) in zip(axes, groups):
            unresolved: List[str] = []
            nonmonotonic_parameters: List[str] = []
            for parameter in parameters:
                profile = targets["profiles"][parameter]
                finite_points = [
                    point
                    for point in profile["profile_points"]
                    if point["probability"] is not None
                ]
                finite_points.sort(key=lambda point: point["multiplier"])
                multiplier = np.asarray([point["multiplier"] for point in finite_points])
                probability = np.asarray([point["probability"] for point in finite_points])
                is_cov = parameter.startswith("r_c")
                label = "COV multiplier" if is_cov else "mean multiplier"
                if profile["observed_nonmonotonic"]:
                    nonmonotonic_parameters.append(parameter)
                color = colors["cov" if is_cov else "mean"]
                ax.plot(multiplier, probability, marker="o", markersize=2.2, lw=1.4, color=color, label=label)
                for root in profile["roots"]:
                    ax.scatter(
                        root["multiplier"],
                        root["probability"],
                        marker="*",
                        s=90,
                        color=color,
                        edgecolor="black",
                        linewidth=0.4,
                        zorder=5,
                    )
                if not profile["crossing_exists"]:
                    unresolved.append(parameter)
            ax.axvline(1.0, color="0.35", linestyle=":", linewidth=1.0, label="baseline multiplier")
            ax.axhline(TARGET_REFERENCE_PROBABILITY, color="#4daf4a", linestyle="--", linewidth=1.1, label="0.10 criterion")
            ax.axhline(TARGET_PROBABILITY, color="#e41a1c", linestyle="--", linewidth=1.1, label="0.05 criterion")
            ax.scatter([1.0], [DIRECT_PROBABILITY_REQUIRED], color="black", s=24, zorder=6)
            ax.set_xscale("log")
            ax.set_title(panel_title)
            ax.set_xlabel("Synthetic parameter multiplier")
            ax.grid(True, which="both", alpha=0.25)
            panel_notes: List[str] = []
            if unresolved:
                panel_notes.append("No resolved 0.05 crossing: " + ", ".join(unresolved))
            if nonmonotonic_parameters:
                panel_notes.append(
                    "Observed nonmonotonic: " + ", ".join(nonmonotonic_parameters)
                )
            if panel_notes:
                ax.text(
                    0.03,
                    0.03,
                    "\n".join(panel_notes),
                    transform=ax.transAxes,
                    fontsize=7.5,
                    va="bottom",
                )
        axes[0].set_ylabel(r"Authoritative direct $P(DV>1)$")
        axes[2].set_xticks([0.7, 1.0, 1.4])
        axes[2].set_xticklabels(["0.7", "1.0", "1.4"])
        axes[2].minorticks_off()
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.suptitle(
            "Synthetic target-oriented parameter scenarios",
            y=0.995,
            fontsize=14,
        )
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=5,
            frameon=False,
        )
        fig.text(
            0.5,
            -0.01,
            "Illustrative thresholds; not an agricultural standard, intervention, or validation result.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.86))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_uq_target_extension(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> Dict[str, Any]:
    """Run the complete UQ and target analysis and write machine-readable outputs."""

    hashes_before = protected_hashes()
    hash_check_before = verify_protected_hashes(hashes_before)
    if not hash_check_before["all_match"]:
        raise RuntimeError(f"Protected hash gate failed: {hash_check_before['mismatches']}")

    direct_probability = authoritative_direct_probability()
    baseline_scenario_probability = float(direct_scenario_probability())
    if baseline_scenario_probability != direct_probability:
        raise RuntimeError(
            "Generalized target-scenario propagation does not reproduce the protected baseline exactly."
        )

    monte_carlo = monte_carlo_verification()
    grid_refinement_required = bool(
        abs(float(monte_carlo["probability"]) - direct_probability)
        > 1.96 * float(monte_carlo["bernoulli_standard_error"])
    )
    grid_refinement = grid_refinement_diagnostic() if grid_refinement_required else []
    form = run_form_analysis()
    targets = run_all_target_profiles()
    comparison = probability_comparison(direct_probability, monte_carlo, form)

    output_directory.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_directory / "pbfe_uq_target_diagnostics.json"
    table_path = output_directory / "pbfe_uq_target_sensitivity_verification_table.csv"
    target_csv_path = output_directory / "pbfe_uq_target_profiles_and_roots.csv"

    write_sensitivity_table(table_path, direct_probability, monte_carlo, form)
    write_target_csv(target_csv_path, targets)

    hashes_after = protected_hashes()
    hash_check_after = verify_protected_hashes(hashes_after)
    if not hash_check_after["all_match"] or hashes_after != hashes_before:
        raise RuntimeError("A protected baseline artifact changed during extension execution.")

    diagnostics: Dict[str, Any] = {
        "status": "synthetic numerical extension; not empirical validation",
        "scope": {
            "fixed_environmental_state": "D1 with probability one; not annual frequency",
            "continuous_hierarchy_role": "FORM and independent Monte Carlo diagnostics",
            "authoritative_probability_role": "existing discrete midpoint-bin propagation",
            "target_scenarios": "synthetic parameter scenarios, not interventions",
            "directional_cosines": "local at DV=1, not global sensitivity indices",
            "economic_or_utility_analysis": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "protected_artifacts": {
            "expected_sha256": dict(EXPECTED_PROTECTED_SHA256),
            "sha256_before": hashes_before,
            "sha256_after": hashes_after,
            "before_check": hash_check_before,
            "after_check": hash_check_after,
            "byte_identical_during_run": hashes_before == hashes_after,
        },
        "direct_discrete": {
            "probability": direct_probability,
            "authoritative": True,
            "generalized_scenario_baseline_probability": baseline_scenario_probability,
            "exact_recovery": baseline_scenario_probability == direct_probability,
        },
        "continuous_monte_carlo": monte_carlo,
        "grid_refinement": {
            "triggered": grid_refinement_required,
            "trigger_rule": "abs(MC-direct) > 1.96*MC standard error",
            "records": grid_refinement,
            "diagnostic_only": True,
        },
        "form": form,
        "probability_comparison": comparison,
        "target_analysis": targets,
        "outputs": {
            "diagnostics_json": diagnostics_path.relative_to(PROJECT_ROOT).as_posix(),
            "sensitivity_verification_csv": table_path.relative_to(PROJECT_ROOT).as_posix(),
            "target_profiles_and_roots_csv": target_csv_path.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    diagnostics_path.write_text(
        json.dumps(_as_builtin(diagnostics), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return diagnostics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for machine-readable UQ and target-extension outputs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    diagnostics = run_uq_target_extension(arguments.output_dir)
    monte_carlo = diagnostics["continuous_monte_carlo"]
    form = diagnostics["form"]
    print(f"Direct authoritative P(DV>1): {diagnostics['direct_discrete']['probability']!r}")
    print(
        "Continuous Monte Carlo: "
        f"count={monte_carlo['exceedance_count']}, "
        f"N={monte_carlo['sample_size']}, "
        f"p={monte_carlo['probability']!r}, "
        f"SE={monte_carlo['bernoulli_standard_error']!r}"
    )
    print(
        f"FORM: beta={form['beta']!r}, p={form['form_probability']!r}, "
        f"u*={form['governing_design_point']}, alpha={form['alpha']}"
    )
    print(f"Outputs: {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
