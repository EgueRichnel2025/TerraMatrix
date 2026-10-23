"""Reproducible least-squares solver comparison experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from collections.abc import Callable, Sequence
from math import isfinite
from numbers import Real
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import qr as scipy_qr

from terramatrix.numerical import (
    diagnose_matrix,
    solve_normal_equations,
    solve_pivoted_qr,
    solve_qr,
    solve_svd,
)
from terramatrix.reproducibility import collect_environment_metadata

DEFAULT_OUTPUT = Path("results/experiments/least_squares_comparison.json")
DEFAULT_RANDOM_SEED = 20261022
DEFAULT_TARGET_CONDITION_NUMBERS = (1e2, 1e4, 1e6)

Solver = Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]


def _array_sha256(values: NDArray[np.float64]) -> str:
    """Hash an array's shape, float64 representation, and numerical values."""
    normalized = np.asarray(values, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(b"\\0float64\\0")
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _validate_integer(value: int, name: str, minimum: int) -> int:
    """Validate an integer parameter, rejecting booleans."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(
            f"'{name}' must be an integer greater than or equal to {minimum}."
        )

    return int(value)


def _validate_real(
    value: Real,
    name: str,
    minimum: float,
) -> float:
    """Validate a finite real parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"'{name}' must be a finite real scalar.")

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{name}' must be a finite real scalar.") from exc

    if not np.isfinite(result) or result < minimum:
        raise ValueError(
            f"'{name}' must be finite and greater than or equal to {minimum}."
        )

    return result


def _validate_condition_numbers(
    values: Sequence[Real],
) -> tuple[float, ...]:
    """Validate target condition numbers for the ill-conditioned family."""
    try:
        targets = tuple(
            _validate_real(value, "target_condition_numbers entry", 1.0)
            for value in values
        )
    except TypeError as exc:
        raise ValueError(
            "'target_condition_numbers' must be a non-empty sequence."
        ) from exc

    if not targets:
        raise ValueError("'target_condition_numbers' must not be empty.")

    return targets


def _controlled_spectrum_matrix(
    n_samples: int,
    n_features: int,
    singular_values: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Construct a matrix with a prescribed singular spectrum."""
    left_random = rng.standard_normal((n_samples, n_features))
    left_vectors, _ = np.linalg.qr(left_random, mode="reduced")

    right_random = rng.standard_normal((n_features, n_features))
    right_vectors, _ = np.linalg.qr(right_random)

    matrix = (left_vectors * singular_values) @ right_vectors.T

    if not np.isfinite(matrix).all():
        raise FloatingPointError("Matrix generation produced non-finite values.")

    return np.asfortranarray(matrix, dtype=np.float64)


def _generate_case_matrix(
    case_name: str,
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
    target_condition_number: float | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Generate one matrix using the documented experimental construction."""
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
            "algorithm": "orthogonal_factors_with_prescribed_spectrum",
            "prescribed_singular_values": singular_values.tolist(),
            "prescribed_condition_number_2": 2.0,
        }

    elif case_name == "ill_conditioned":
        if target_condition_number is None:
            raise ValueError(
                "'target_condition_number' is required for this case."
            )

        singular_values = np.geomspace(
            1.0,
            1.0 / target_condition_number,
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
            "algorithm": "orthogonal_factors_with_prescribed_spectrum",
            "prescribed_singular_values": singular_values.tolist(),
            "prescribed_condition_number_2": target_condition_number,
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


def _condition_label(value: float) -> str:
    """Create a compact, stable label for a target condition number."""
    return (
        f"{value:.0e}"
        .replace("e+0", "e")
        .replace("e+", "e")
        .replace("e-0", "e-")
    )


def _run_attempt(
    solver: Solver,
    matrix: NDArray[np.float64],
    response: NDArray[np.float64],
) -> tuple[dict[str, Any], NDArray[np.float64] | None]:
    """Execute one solver attempt, retaining timing, warnings, and failures."""
    error: Exception | None = None
    coefficients: NDArray[np.float64] | None = None

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        started = perf_counter_ns()

        try:
            result = solver(matrix, response)
        except Exception as exc:
            error = exc
            result = None

        elapsed_seconds = (perf_counter_ns() - started) / 1e9

    recorded_warnings = [
        {
            "category": type(item.message).__name__,
            "message": str(item.message),
        }
        for item in captured_warnings
    ]

    if error is None:
        try:
            coefficients = np.asarray(result, dtype=np.float64)

            if coefficients.shape != (matrix.shape[1],):
                raise ValueError(
                    "The solver returned coefficients with an invalid shape."
                )

            if not np.isfinite(coefficients).all():
                raise FloatingPointError(
                    "The solver returned non-finite coefficients."
                )

        except Exception as exc:
            error = exc
            coefficients = None

    successful = error is None and coefficients is not None

    trial = {
        "status": "success" if successful else "failure",
        "elapsed_seconds": float(elapsed_seconds),
        "warnings": recorded_warnings,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }

    return trial, coefficients if successful else None


def _pivoted_qr_rank(
    matrix: NDArray[np.float64],
) -> dict[str, Any]:
    """Estimate rank using the diagonal of column-pivoted R."""
    _, r, _ = scipy_qr(
        matrix,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )

    diagonal = np.abs(np.diag(r))
    scale = float(diagonal[0]) if diagonal.size else 0.0
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * scale

    rank = 0
    for value in diagonal:
        if value <= tolerance:
            break
        rank += 1

    return {
        "method": "column_pivoted_qr_diagonal",
        "estimated_rank": rank,
        "rank_tolerance": float(tolerance),
    }


def _method_rank_diagnostics(
    method_name: str,
    matrix: NDArray[np.float64],
    matrix_diagnostics: Any,
) -> dict[str, Any]:
    """Record the rank estimate relevant to each solver."""
    if method_name == "svd_pseudoinverse":
        return {
            "method": "svd_singular_values",
            "estimated_rank": matrix_diagnostics.numerical_rank,
            "rank_tolerance": matrix_diagnostics.rank_tolerance,
        }

    if method_name == "qr_pivoted":
        return _pivoted_qr_rank(matrix)

    return {
        "method": "not_estimated_by_solver",
        "estimated_rank": None,
        "rank_tolerance": None,
    }


def _solution_metrics(
    matrix: NDArray[np.float64],
    response: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    beta_true: NDArray[np.float64],
    *,
    coefficient_error_meaningful: bool,
    matrix_diagnostics: Any,
) -> dict[str, Any]:
    """Calculate residual, coefficient, and least-squares optimality metrics."""
    residual = matrix @ coefficients - response
    residual_norm = float(np.linalg.norm(residual))
    response_norm = float(np.linalg.norm(response))
    gradient_norm = float(np.linalg.norm(matrix.T @ residual))

    matrix_norm_2 = float(matrix_diagnostics.largest_singular_value)
    stationarity_scale = matrix_norm_2 * residual_norm

    if stationarity_scale > 0.0:
        relative_stationarity = gradient_norm / stationarity_scale
    else:
        relative_stationarity = 0.0 if gradient_norm == 0.0 else None

    if response_norm > 0.0:
        relative_residual = residual_norm / response_norm
    else:
        relative_residual = 0.0 if residual_norm == 0.0 else None

    if coefficient_error_meaningful:
        true_norm = float(np.linalg.norm(beta_true))
        coefficient_error = float(
            np.linalg.norm(coefficients - beta_true) / true_norm
        ) if true_norm > 0.0 else None
    else:
        coefficient_error = None

    return {
        "coefficients": coefficients.tolist(),
        "coefficient_norm_2": float(np.linalg.norm(coefficients)),
        "reference_coefficient_norm_2": float(np.linalg.norm(beta_true)),
        "residual_norm_2": residual_norm,
        "relative_residual": relative_residual,
        "relative_coefficient_error_2": coefficient_error,
        "least_squares_gradient_norm_2": gradient_norm,
        "relative_stationarity_measure": relative_stationarity,
    }


def _run_method(
    method_name: str,
    solver: Solver,
    matrix: NDArray[np.float64],
    response: NDArray[np.float64],
    beta_true: NDArray[np.float64],
    *,
    coefficient_error_meaningful: bool,
    matrix_diagnostics: Any,
    warmup_runs: int,
    timed_repetitions: int,
) -> dict[str, Any]:
    """Run one method, preserving all attempts and summarizing valid timings."""
    warmup_trials: list[dict[str, Any]] = []
    timed_trials: list[dict[str, Any]] = []
    successful_solutions: list[NDArray[np.float64]] = []

    for _ in range(warmup_runs):
        trial, coefficients = _run_attempt(solver, matrix, response)
        warmup_trials.append(trial)
        if coefficients is not None:
            successful_solutions.append(coefficients)

    for _ in range(timed_repetitions):
        trial, coefficients = _run_attempt(solver, matrix, response)
        timed_trials.append(trial)
        if coefficients is not None:
            successful_solutions.append(coefficients)

    all_trials = warmup_trials + timed_trials
    successful_count = sum(
        trial["status"] == "success" for trial in all_trials
    )

    if successful_count == len(all_trials):
        solver_status = "success"
    elif successful_count == 0:
        solver_status = "failure"
    else:
        solver_status = "partial_failure"

    all_trials_succeeded = solver_status == "success"
    raw_times = [
        float(trial["elapsed_seconds"]) for trial in timed_trials
    ]

    if all_trials_succeeded:
        median_seconds = float(np.median(raw_times))
        iqr_seconds = float(
            np.percentile(raw_times, 75) - np.percentile(raw_times, 25)
        )
        timing_status = "complete"
    else:
        median_seconds = None
        iqr_seconds = None
        timing_status = "incomplete_due_to_solver_failure"

    warning_items: list[dict[str, str]] = []
    seen_warnings: set[tuple[str, str]] = set()

    for trial in all_trials:
        for item in trial["warnings"]:
            key = (item["category"], item["message"])
            if key not in seen_warnings:
                warning_items.append(item)
                seen_warnings.add(key)

    metrics = None
    if successful_solutions:
        metrics = _solution_metrics(
            matrix,
            response,
            successful_solutions[-1],
            beta_true,
            coefficient_error_meaningful=coefficient_error_meaningful,
            matrix_diagnostics=matrix_diagnostics,
        )

    return {
        "solver_status": solver_status,
        "solver_rank_diagnostics": _method_rank_diagnostics(
            method_name,
            matrix,
            matrix_diagnostics,
        ),
        "metrics": metrics,
        "warnings": warning_items,
        "warmup_trials": warmup_trials,
        "timing": {
            "performed": True,
            "status": timing_status,
            "warmup_runs": warmup_runs,
            "timed_repetitions": timed_repetitions,
            "raw_execution_times_seconds": raw_times,
            "median_seconds": median_seconds,
            "interquartile_range_seconds": iqr_seconds,
            "summary_note": (
                "Summary statistics are only reported when every warm-up "
                "and timed solver attempt succeeds. Failed-attempt durations "
                "are retained but are not presented as comparable runtimes."
            ),
            "timed_trials": timed_trials,
        },
    }


def _run_case(
    *,
    experiment_id: str,
    case_name: str,
    seed: int,
    n_samples: int,
    n_features: int,
    noise_std: float,
    warmup_runs: int,
    timed_repetitions: int,
    target_condition_number: float | None = None,
) -> dict[str, Any]:
    """Run every solver on identical input data for one experiment."""
    rng = np.random.default_rng(seed)

    matrix, generation = _generate_case_matrix(
        case_name,
        n_samples,
        n_features,
        rng,
        target_condition_number,
    )

    beta_true = rng.standard_normal(n_features)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    response = matrix @ beta_true + noise

    if not (
        np.isfinite(matrix).all()
        and np.isfinite(beta_true).all()
        and np.isfinite(noise).all()
        and np.isfinite(response).all()
    ):
        raise FloatingPointError("Generated experimental data are not finite.")

    diagnostics = diagnose_matrix(matrix)

    expected_rank = n_features - 1 if case_name == "rank_deficient" else n_features

    if diagnostics.numerical_rank != expected_rank:
        raise AssertionError(
            f"{experiment_id}: expected numerical rank {expected_rank}, "
            f"observed {diagnostics.numerical_rank}."
        )

    if isfinite(diagnostics.condition_number_2):
        condition_number: float | None = float(diagnostics.condition_number_2)
        condition_status = "finite"
    else:
        condition_number = None
        condition_status = "infinite_due_to_numerical_rank_deficiency"

    solver_definitions: tuple[tuple[str, Solver], ...] = (
        ("normal_equations", solve_normal_equations),
        ("qr_unpivoted", solve_qr),
        ("qr_pivoted", solve_pivoted_qr),
        ("svd_pseudoinverse", solve_svd),
    )

    methods = {
        name: _run_method(
            name,
            solver,
            matrix,
            response,
            beta_true,
            coefficient_error_meaningful=case_name != "rank_deficient",
            matrix_diagnostics=diagnostics,
            warmup_runs=warmup_runs,
            timed_repetitions=timed_repetitions,
        )
        for name, solver in solver_definitions
    }

    return {
        "experiment_id": experiment_id,
        "experiment_family": (
            "TM-LS-02" if case_name == "ill_conditioned" else experiment_id
        ),
        "case": case_name,
        "random_seed": seed,
        "matrix_generation": generation,
        "noise": {
            "standard_deviation": noise_std,
            "measured_l2_norm": float(np.linalg.norm(noise)),
            "sha256": _array_sha256(noise),
        },
        "reference_coefficients": beta_true.tolist(),
        "reference_coefficients_sha256": _array_sha256(beta_true),
        "response_sha256": _array_sha256(response),
        "matrix": {
            "shape": list(matrix.shape),
            "dtype": "float64",
            "frobenius_norm": float(np.linalg.norm(matrix, ord="fro")),
            "sha256": _array_sha256(matrix),
            "response_l2_norm": float(np.linalg.norm(response)),
        },
        "rank_diagnostics": {
            "numerical_rank": diagnostics.numerical_rank,
            "maximum_rank": diagnostics.maximum_rank,
            "expected_numerical_rank": expected_rank,
            "expectation_matches": diagnostics.numerical_rank == expected_rank,
            "rank_tolerance": diagnostics.rank_tolerance,
            "largest_singular_value": diagnostics.largest_singular_value,
            "smallest_singular_value": diagnostics.smallest_singular_value,
            "target_condition_number_2": target_condition_number,
            "measured_condition_number_2": condition_number,
            "condition_number_status": condition_status,
        },
        "methods": methods,
    }


def run_least_squares_comparison_experiments(
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    n_samples: int = 1000,
    n_features: int = 20,
    seed: int = DEFAULT_RANDOM_SEED,
    noise_std: float = 1e-3,
    target_condition_numbers: Sequence[Real] = (
        DEFAULT_TARGET_CONDITION_NUMBERS
    ),
    warmup_runs: int = 1,
    timed_repetitions: int = 5,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run and persist the TM-LS comparison suite with actual measurements."""
    n_samples = _validate_integer(n_samples, "n_samples", 2)
    n_features = _validate_integer(n_features, "n_features", 2)
    seed = _validate_integer(seed, "seed", 0)
    warmup_runs = _validate_integer(warmup_runs, "warmup_runs", 0)
    timed_repetitions = _validate_integer(
        timed_repetitions,
        "timed_repetitions",
        1,
    )
    noise_std = _validate_real(noise_std, "noise_std", 0.0)
    target_condition_numbers = _validate_condition_numbers(
        target_condition_numbers
    )

    if n_samples <= n_features:
        raise ValueError(
            "'n_samples' must be greater than 'n_features' for these "
            "overdetermined full-column-rank experiments."
        )

    cases: list[tuple[str, str, float | None]] = [
        ("TM-LS-01", "well_conditioned", None)
    ]

    for target in target_condition_numbers:
        cases.append(
            (
                f"TM-LS-02-kappa-{_condition_label(target)}",
                "ill_conditioned",
                target,
            )
        )

    cases.append(("TM-LS-03", "rank_deficient", None))

    environment = collect_environment_metadata(project_root=project_root)

    experiments = [
        _run_case(
            experiment_id=experiment_id,
            case_name=case_name,
            seed=seed,
            n_samples=n_samples,
            n_features=n_features,
            noise_std=noise_std,
            warmup_runs=warmup_runs,
            timed_repetitions=timed_repetitions,
            target_condition_number=target,
        )
        for experiment_id, case_name, target in cases
    ]

    report: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": "TM-LS-COMPARISON-01",
        "purpose": "least_squares_numerical_accuracy_and_timing",
        "performance_measurements": {
            "performed": True,
            "warmup_runs": warmup_runs,
            "timed_repetitions": timed_repetitions,
            "timer": "time.perf_counter_ns",
            "timing_scope": (
                "Public solver call on preconstructed inputs; includes "
                "solver input validation and memory preparation, but excludes "
                "matrix generation and post-solve metric calculations."
            ),
            "interpretation": (
                "Per-method timing summaries are omitted when a solver "
                "fails in any warm-up or timed attempt."
            ),
        },
        "parameters": {
            "n_samples": n_samples,
            "n_features": n_features,
            "dtype": "float64",
            "base_random_seed": seed,
            "case_seed_policy": (
                "The base seed is restarted independently for each case. "
                "Within the controlled-spectrum family, identical random "
                "factors, reference coefficients, and noise are reused "
                "across target condition numbers."
            ),
            "noise_standard_deviation": noise_std,
            "target_condition_numbers": list(target_condition_numbers),
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
    """Command-line entry point for the least-squares comparison suite."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare least-squares solvers on reproducible synthetic matrices."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--noise-std", type=float, default=1e-3)
    parser.add_argument(
        "--target-condition-numbers",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_CONDITION_NUMBERS),
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-repetitions", type=int, default=5)
    parser.add_argument("--project-root", type=Path, default=None)
    arguments = parser.parse_args(argv)

    report = run_least_squares_comparison_experiments(
        output_path=arguments.output,
        n_samples=arguments.n_samples,
        n_features=arguments.n_features,
        seed=arguments.seed,
        noise_std=arguments.noise_std,
        target_condition_numbers=arguments.target_condition_numbers,
        warmup_runs=arguments.warmup_runs,
        timed_repetitions=arguments.timed_repetitions,
        project_root=arguments.project_root,
    )

    print(f"Report written to: {arguments.output}")
    print(
        f"Experiments: {len(report['experiments'])}; "
        f"timed repetitions per method: {arguments.timed_repetitions}"
    )

    for experiment in report["experiments"]:
        rank = experiment["rank_diagnostics"]["numerical_rank"]
        methods = experiment["methods"]
        statuses = ", ".join(
            f"{name}={metrics['solver_status']}"
            for name, metrics in methods.items()
        )
        print(
            f"{experiment['experiment_id']}: "
            f"rank={rank}/{arguments.n_features}; {statuses}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
