"""Generate the publication-facing PBFE target-oriented figure.

The display grid is intentionally separate from the extended inverse-target
search recorded by the UQ extension. This script reuses the verified authoritative
discrete scenario evaluator over the fixed journal display range 0.5 <= r <=
2.0; it does not modify or replace any extension result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np

import pbfe_uq_target_extension as extension


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DEFAULT_DIAGNOSTICS = (
    PROJECT_ROOT
    / "outputs"
    / "uq_target_extension"
    / "pbfe_uq_target_diagnostics.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "pbfe_target_scenarios.png"
DISPLAY_MINIMUM = 0.5
DISPLAY_MAXIMUM = 2.0
DISPLAY_POINT_COUNT = 151


PANEL_PARAMETERS = (
    ("(a) LAI", ("r_mL", "r_cL")),
    ("(b) TFP link", ("r_mT", "r_cT")),
    ("(c) Normalized-loss link", ("r_mD", "r_cD")),
)


def load_verified_roots(path: Path) -> Dict[str, list[Mapping[str, float]]]:
    """Read the independently verified extension roots and status metadata."""

    with path.open(encoding="utf-8") as stream:
        diagnostics = json.load(stream)
    direct = float(diagnostics["direct_discrete"]["probability"])
    if direct != extension.DIRECT_PROBABILITY_REQUIRED:
        raise RuntimeError("Extension diagnostics do not match the protected direct result.")
    return {
        parameter: list(profile["roots"])
        for parameter, profile in diagnostics["target_analysis"]["profiles"].items()
    }


def evaluate_display_grid(multipliers: np.ndarray) -> Dict[str, np.ndarray]:
    """Evaluate every scenario directly on the deterministic display grid."""

    return {
        parameter: np.asarray(
            [
                extension.direct_scenario_probability({parameter: float(multiplier)})
                for multiplier in multipliers
            ],
            dtype=float,
        )
        for parameter in extension.TARGET_PARAMETER_METADATA
    }


def create_figure(
    output_path: Path,
    diagnostics_path: Path = DEFAULT_DIAGNOSTICS,
    display_minimum: float = DISPLAY_MINIMUM,
    display_maximum: float = DISPLAY_MAXIMUM,
    point_count: int = DISPLAY_POINT_COUNT,
) -> None:
    """Create the deterministic three-panel journal figure."""

    if not 0.0 < display_minimum < 1.0 < display_maximum:
        raise ValueError("The positive display interval must contain the baseline r=1.")
    if point_count < 101:
        raise ValueError("Use at least 101 points for the journal display grid.")

    multipliers = np.linspace(display_minimum, display_maximum, point_count)
    roots = load_verified_roots(diagnostics_path)
    profiles = evaluate_display_grid(multipliers)

    colors = {"mean": "#1f77b4", "cov": "#d95f02"}
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55), sharex=True)
    for axis, (panel_title, parameters) in zip(axes, PANEL_PARAMETERS):
        panel_values = []
        for parameter in parameters:
            is_cov = parameter.startswith("r_c")
            label = "COV multiplier" if is_cov else "Nominal-mean multiplier"
            values = profiles[parameter]
            panel_values.append(values)
            axis.plot(
                multipliers,
                values,
                color=colors["cov" if is_cov else "mean"],
                linewidth=1.8,
                label=label,
            )
            for root in roots[parameter]:
                root_multiplier = float(root["multiplier"])
                if display_minimum <= root_multiplier <= display_maximum:
                    axis.scatter(
                        root_multiplier,
                        float(root["probability"]),
                        marker="*",
                        s=85,
                        color=colors["cov" if is_cov else "mean"],
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=6,
                    )

        axis.axvline(1.0, color="0.35", linestyle=":", linewidth=1.1)
        axis.axhline(
            extension.TARGET_REFERENCE_PROBABILITY,
            color="#4daf4a",
            linestyle="--",
            linewidth=1.2,
        )
        axis.axhline(
            extension.TARGET_PROBABILITY,
            color="#e41a1c",
            linestyle="--",
            linewidth=1.2,
        )
        axis.scatter(
            [1.0],
            [extension.DIRECT_PROBABILITY_REQUIRED],
            color="black",
            s=25,
            zorder=7,
        )
        axis.set_title(panel_title, fontsize=10.5)
        axis.set_xlim(display_minimum, display_maximum)
        axis.set_xticks([0.5, 1.0, 1.5, 2.0])
        axis.set_xlabel(r"Synthetic parameter multiplier, $r$", fontsize=9)
        axis.grid(True, alpha=0.22)
        axis.tick_params(labelsize=8.5)

        values_for_limits = np.concatenate(panel_values)
        lower = min(float(np.min(values_for_limits)), extension.TARGET_PROBABILITY)
        upper = max(float(np.max(values_for_limits)), extension.TARGET_REFERENCE_PROBABILITY)
        padding = 0.06 * max(upper - lower, 0.05)
        axis.set_ylim(max(0.0, lower - padding), min(1.0, upper + padding))

    axes[0].set_ylabel(r"Direct $P(DV>1.00)$", fontsize=9)
    axes[0].text(
        0.03,
        0.04,
        r"$r_{cL}$: no resolved 5% crossing; nonmonotonic",
        transform=axes[0].transAxes,
        fontsize=7.1,
    )
    axes[1].text(
        0.03,
        0.04,
        r"$r_{cT}$: no resolved 5% crossing",
        transform=axes[1].transAxes,
        fontsize=7.1,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    threshold_handles = [
        plt.Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.1),
        plt.Line2D([0], [0], color="#4daf4a", linestyle="--", linewidth=1.2),
        plt.Line2D([0], [0], color="#e41a1c", linestyle="--", linewidth=1.2),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="white",
            markerfacecolor="0.35",
            markeredgecolor="black",
            markersize=8,
        ),
    ]
    fig.legend(
        handles + threshold_handles,
        labels + ["Baseline $r=1$", "$p=0.10$", "$p=0.05$", "Resolved 5% crossing"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=8.1,
    )
    fig.text(
        0.5,
        0.005,
        "Panel-specific vertical scales; direct discrete evaluation on a 151-point display grid.",
        ha="center",
        fontsize=7.8,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.84), w_pad=1.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--display-minimum", type=float, default=DISPLAY_MINIMUM)
    parser.add_argument("--display-maximum", type=float, default=DISPLAY_MAXIMUM)
    parser.add_argument("--point-count", type=int, default=DISPLAY_POINT_COUNT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    create_figure(
        output_path=arguments.output,
        diagnostics_path=arguments.diagnostics,
        display_minimum=arguments.display_minimum,
        display_maximum=arguments.display_maximum,
        point_count=arguments.point_count,
    )
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
