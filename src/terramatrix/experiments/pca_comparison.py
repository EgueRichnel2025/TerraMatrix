"""Reproducible numerical comparison of covariance-based and SVD-based PCA."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from terramatrix.numerical import (
    diagnose_matrix,
    principal_component_analysis,
)
from terramatrix.reproducibility import collect_environment_metadata

DEFAULT_OUTPUT = Path("results/experiments/pca_comparison.json")
DEFAULT_RANDOM_SEED = 20261022


def _validate_integer(value: int, name: str, minimum: int) -> int:
    """Validate a non-negative or positive integer parameter."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(
            f"'{name}' must be an integer greater than or equal to {minimum}."
        )

    return int(value)


def _controlled_spectrum_matrix(
    n_samples: int,
    n_features: int,
    singular_values: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Build an approximately centered matrix with a prescribed spectrum."""
    left_random = rng.standard_normal((n_samples, n_features))
    left_random -= np.mean(left_random, axis=0, keepdims=True)

    left_vectors, _ = np.linalg.qr(left_random, mode="reduced")

    right_random = rng.standard_normal((n_features, n_features))
    right_vectors, _ = np.linalg.qr(right_random)

    matrix = (left_vectors * singular_values) @ right_vectors.T

    if not np.isfinite(matrix).all():
        raise FloatingPointError(
            "Synthetic matrix generation produced non-finite values."
        )

    return np.asarray(matrix, dtype=np.float64)


def _generate_case_matrix(
    case_name: str,
    n_samples: int,
    n_features: int,
    seed: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Generate one deterministic experimental matrix."""
    rng = np.random.default_rng(seed)

    if case_name == "well_conditioned":
        singular_values = np.linspace(
            1.0,
            0.5,
            n_features,
            dtype=np.float64,
        )
        matrix = _controlled_spectrum_matrix(
            n_samples,
            n_features,
            singular_values,
            rng,
        )
        generation = {
            "algorithm": "centered_orthogonal_spectrum",
            "prescribed_singular_values": singular_values.tolist(),
            "prescribed_condition_number_2": 2.0,
        }

    elif case_name == "ill_conditioned":
        singular_values = np.geomspace(
            1.0,
            1e-6,
            n_features,
            dtype=np.float64,
        )
        matrix = _controlled_spectrum_matrix(
            n_samples,
            n_features,
            singular_values,
            rng,
        )
        generation = {
            "algorithm": "centered_orthogonal_spectrum",
            "prescribed_singular_values": singular_values.tolist(),
            "prescribed_condition_number_2": 1e6,
        }

    elif case_name == "rank_deficient":
        matrix = rng.standard_normal((n_samples, n_features))
        matrix[:, -1] = matrix[:, 0]

        if not np.array_equal(matrix[:, -1], matrix[:, 0]):
            raise RuntimeError("The duplicate columns were not constructed exactly.")

        generation = {
            "algorithm": "random_matrix_with_exact_duplicate_column",
            "duplicate_column_indices_zero_based": [0, n_features - 1],
            "collinearity_epsilon": 0.0,
        }

    else:
        raise ValueError(f"Unknown experiment case: {case_name!r}.")

    return np.asfortranarray(matrix), generation


def _relative_error(
    reference: NDArray[np.float64],
    approximation: NDArray[np.float64],
) -> float:
    """Calculate a relative Frobenius error, handling a zero reference."""
    numerator = float(np.linalg.norm(reference - approximation, ord="fro"))
    denominator = float(np.linalg.norm(reference, ord="fro"))

    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")

    return numerator / denominator


def _summarize_method(
    matrix: NDArray[np.float64],
    centered: NDArray[np.float64],
    method: str,
) -> dict[str, Any]:
    """Run PCA and calculate numerical validation metrics."""
    result = principal_component_analysis(
        matrix,
        method=method,
    )

    reconstructed_centered = result.scores @ result.components
    identity = np.eye(result.n_components, dtype=np.float64)

    reconstruction_error = _relative_error(
        centered,
        reconstructed_centered,
    )

    orthogonality_error = float(
        np.linalg.norm(
            result.components @ result.components.T - identity,
            ord="fro",
        )
    )

    metrics = {
        "n_components": result.n_components,
        "singular_values": result.singular_values.tolist(),
        "explained_variance": result.explained_variance.tolist(),
        "explained_variance_ratio": result.explained_variance_ratio.tolist(),
        "total_variance": float(result.total_variance),
        "reconstruction_relative_error_fro": reconstruction_error,
        "component_orthogonality_error_fro": orthogonality_error,
        "covariance_eigenvalue_tolerance": (
            result.covariance_eigenvalue_tolerance
        ),
    }

    numeric_values = [
        metrics["total_variance"],
        metrics["reconstruction_relative_error_fro"],
        metrics["component_orthogonality_error_fro"],
        *metrics["singular_values"],
        *metrics["explained_variance"],
        *metrics["explained_variance_ratio"],
    ]

    if not all(isfinite(value) for value in numeric_values):
        raise FloatingPointError(
            f"PCA method {method!r} produced non-finite metrics."
        )

    return metrics


def _compare_method_results(
    covariance: dict[str, Any],
    direct_svd: dict[str, Any],
) -> dict[str, float]:
    """Measure agreement between covariance PCA and direct-SVD PCA."""
    covariance_variance = np.asarray(
        covariance["explained_variance"],
        dtype=np.float64,
    )
    svd_variance = np.asarray(
        direct_svd["explained_variance"],
        dtype=np.float64,
    )

    variance_difference = float(
        np.linalg.norm(covariance_variance - svd_variance)
    )
    reference_variance_norm = float(np.linalg.norm(svd_variance))

    relative_variance_difference = (
        variance_difference / reference_variance_norm
        if reference_variance_norm > 0.0
        else variance_difference
    )

    covariance_total = float(covariance["total_variance"])
    svd_total = float(direct_svd["total_variance"])

    relative_total_variance_difference = (
        abs(covariance_total - svd_total) / abs(svd_total)
        if svd_total != 0.0
        else abs(covariance_total - svd_total)
    )

    metrics = {
        "relative_explained_variance_difference_l2": (
            relative_variance_difference
        ),
        "relative_total_variance_difference": (
            relative_total_variance_difference
        ),
    }

    if not all(isfinite(value) for value in metrics.values()):
        raise FloatingPointError(
            "The PCA comparison produced non-finite metrics."
        )

    return metrics


def _compare_singular_spectra(
    covariance: dict[str, Any],
    direct_svd: dict[str, Any],
    *,
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Compare singular values component by component against direct SVD."""
    covariance_s = np.asarray(
        covariance["singular_values"],
        dtype=np.float64,
    )
    reference_s = np.asarray(
        direct_svd["singular_values"],
        dtype=np.float64,
    )

    tolerance = (
        max(shape)
        * np.finfo(np.float64).eps
        * float(reference_s[0])
    )
    significant = reference_s > tolerance
    absolute_s = np.abs(covariance_s - reference_s)

    relative_s = [
        float(error / reference) if keep else None
        for error, reference, keep in zip(
            absolute_s,
            reference_s,
            significant,
            strict=True,
        )
    ]

    covariance_variance = np.asarray(
        covariance["explained_variance"],
        dtype=np.float64,
    )
    reference_variance = np.asarray(
        direct_svd["explained_variance"],
        dtype=np.float64,
    )
    absolute_variance = np.abs(covariance_variance - reference_variance)

    relative_variance = [
        float(error / reference) if keep else None
        for error, reference, keep in zip(
            absolute_variance,
            reference_variance,
            significant,
            strict=True,
        )
    ]

    return {
        "reference_method": "svd",
        "reference_singular_value_tolerance": float(tolerance),
        "reference_numerical_rank": int(np.count_nonzero(significant)),
        "covariance_numerical_rank_at_reference_tolerance": int(
            np.count_nonzero(covariance_s > tolerance)
        ),
        "component_is_significant_by_direct_svd": significant.tolist(),
        "absolute_singular_value_differences": absolute_s.tolist(),
        "relative_singular_value_differences": relative_s,
        "absolute_explained_variance_differences": (
            absolute_variance.tolist()
        ),
        "relative_explained_variance_differences": relative_variance,
    }


def _run_case(
    *,
    experiment_id: str,
    case_name: str,
    seed: int,
    n_samples: int,
    n_features: int,
) -> dict[str, Any]:
    """Execute and validate both PCA methods on the same input matrix."""
    matrix, generation = _generate_case_matrix(
        case_name,
        n_samples,
        n_features,
        seed,
    )

    # PCA analyzes centered data, so diagnose that matrix.
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    diagnostics = diagnose_matrix(centered)

    if case_name == "rank_deficient":
        expected_rank = n_features - 1
        target_condition_number: float | None = None
    else:
        expected_rank = n_features
        target_condition_number = (
            2.0 if case_name == "well_conditioned" else 1e6
        )

    if diagnostics.numerical_rank != expected_rank:
        raise AssertionError(
            f"{experiment_id}: expected numerical rank {expected_rank}, "
            f"observed {diagnostics.numerical_rank}."
        )

    method_results = {
        method: _summarize_method(matrix, centered, method)
        for method in ("covariance", "svd")
    }

    condition_number = float(diagnostics.condition_number_2)

    if isfinite(condition_number):
        reported_condition_number: float | None = condition_number
        condition_status = "finite"
    else:
        # JSON cannot represent infinity as a standard numeric value.
        reported_condition_number = None
        condition_status = "infinite_due_to_numerical_rank_deficiency"

    return {
        "experiment_id": experiment_id,
        "case": case_name,
        "random_seed": seed,
        "matrix_generation": generation,
        "matrix": {
            "shape": list(diagnostics.shape),
            "dtype": "float64",
            "n_samples": n_samples,
            "n_features": n_features,
            "frobenius_norm": float(np.linalg.norm(matrix, ord="fro")),
            "centered_frobenius_norm": float(
                np.linalg.norm(centered, ord="fro")
            ),
        },
        "rank_diagnostics": {
            "diagnosed_matrix": "centered_data_matrix",
            "numerical_rank": diagnostics.numerical_rank,
            "maximum_rank": diagnostics.maximum_rank,
            "expected_numerical_rank": expected_rank,
            "expectation_matches": (
                diagnostics.numerical_rank == expected_rank
            ),
            "rank_tolerance": diagnostics.rank_tolerance,
            "largest_singular_value": diagnostics.largest_singular_value,
            "smallest_singular_value": diagnostics.smallest_singular_value,
            "target_condition_number_2": target_condition_number,
            "measured_condition_number_2": reported_condition_number,
            "condition_number_status": condition_status,
        },
        "methods": method_results,
        "comparison": _compare_method_results(
            method_results["covariance"],
            method_results["svd"],
        ),
        "component_comparison": _compare_singular_spectra(
            method_results["covariance"],
            method_results["svd"],
            shape=diagnostics.shape,
        ),
    }


def run_pca_comparison_experiments(
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    n_samples: int = 1000,
    n_features: int = 20,
    seed: int = DEFAULT_RANDOM_SEED,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the three PCA experiments and persist a reproducible JSON report.

    The experiment-specific default seed follows docs/experiments.md.
    Each case receives a distinct, recorded seed: base seed, base seed + 1,
    and base seed + 2.

    This function measures numerical behavior, not execution speed.
    """
    n_samples = _validate_integer(n_samples, "n_samples", 3)
    n_features = _validate_integer(n_features, "n_features", 2)
    seed = _validate_integer(seed, "seed", 0)

    if n_samples <= n_features:
        raise ValueError(
            "'n_samples' must be greater than 'n_features' so the centered "
            "full-rank cases can retain full column rank."
        )

    environment = collect_environment_metadata(project_root=project_root)

    cases = (
        ("TM-PCA-01", "well_conditioned", seed),
        ("TM-PCA-02", "ill_conditioned", seed + 1),
        ("TM-PCA-03", "rank_deficient", seed + 2),
    )

    experiments = [
        _run_case(
            experiment_id=experiment_id,
            case_name=case_name,
            seed=case_seed,
            n_samples=n_samples,
            n_features=n_features,
        )
        for experiment_id, case_name, case_seed in cases
    ]

    report: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": "TM-PCA-COMPARISON-01",
        "purpose": "numerical_accuracy_comparison",
        "performance_measurements": {
            "performed": False,
            "reason": (
                "Timing is deferred to a separate benchmark experiment."
            ),
        },
        "parameters": {
            "n_samples": n_samples,
            "n_features": n_features,
            "dtype": "float64",
            "base_random_seed": seed,
            "case_seed_policy": "base seed plus case index (0, 1, 2)",
            "timed_repetitions": None,
            "warmup_runs": None,
        },
        "environment": environment,
        "experiments": experiments,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for PCA comparison experiments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare covariance-based PCA with direct-SVD PCA "
            "using reproducible synthetic matrices."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: results/experiments/pca_comparison.json).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of observations (default: 1000).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Number of features (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Base random seed (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root used to collect Git metadata.",
    )

    args = parser.parse_args(argv)

    try:
        report = run_pca_comparison_experiments(
            output_path=args.output,
            n_samples=args.n_samples,
            n_features=args.n_features,
            seed=args.seed,
            project_root=args.project_root,
        )
    except (ValueError, OSError, FloatingPointError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))

    print(f"Report written to: {args.output}")
    print("Numerical comparison completed; no timings were collected.")

    for experiment in report["experiments"]:
        rank = experiment["rank_diagnostics"]
        condition = rank["measured_condition_number_2"]
        condition_text = (
            f"{condition:.6g}"
            if condition is not None
            else "infinite (numerically rank-deficient)"
        )

        comparison = experiment["comparison"]
        print(
            f"{experiment['experiment_id']} ({experiment['case']}): "
            f"rank={rank['numerical_rank']}/{rank['maximum_rank']}, "
            f"condition_number_2={condition_text}, "
            "relative_variance_difference="
            f"{comparison['relative_explained_variance_difference_l2']:.3e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
