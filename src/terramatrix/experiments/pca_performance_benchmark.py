"""Reproducible performance benchmark for covariance PCA and direct SVD PCA."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from terramatrix.experiments.pca_comparison import (
    DEFAULT_RANDOM_SEED,
    _generate_case_matrix,
    _run_case,
    _validate_integer,
)
from terramatrix.numerical import principal_component_analysis
from terramatrix.reproducibility import collect_environment_metadata

DEFAULT_OUTPUT = Path("results/experiments/pca_performance.json")

CASES = (
    ("TM-PCA-01", "well_conditioned"),
    ("TM-PCA-02", "ill_conditioned"),
    ("TM-PCA-03", "rank_deficient"),
)
METHODS = ("covariance", "svd")


def _benchmark_method(
    matrix: np.ndarray,
    method: str,
    warmup_runs: int,
    timed_repetitions: int,
) -> dict[str, Any]:
    """Time only the PCA call, excluding matrix generation and validation."""
    for _ in range(warmup_runs):
        principal_component_analysis(matrix, method=method)

    durations = []

    for _ in range(timed_repetitions):
        start = perf_counter_ns()
        principal_component_analysis(matrix, method=method)
        elapsed = perf_counter_ns() - start
        durations.append(elapsed / 1_000_000_000)

    q1, q3 = np.percentile(durations, [25, 75])

    return {
        "timer": "time.perf_counter_ns",
        "warmup_runs": warmup_runs,
        "timed_repetitions": timed_repetitions,
        "raw_execution_times_seconds": durations,
        "median_seconds": statistics.median(durations),
        "mean_seconds": statistics.mean(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "iqr_seconds": float(q3 - q1),
    }


def run_pca_performance_benchmark(
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    n_samples: int = 1000,
    n_features: int = 20,
    seed: int = DEFAULT_RANDOM_SEED,
    warmup_runs: int = 3,
    timed_repetitions: int = 20,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark both PCA methods on three reproducible matrix cases."""
    n_samples = _validate_integer(n_samples, "n_samples", 3)
    n_features = _validate_integer(n_features, "n_features", 2)
    seed = _validate_integer(seed, "seed", 0)
    warmup_runs = _validate_integer(warmup_runs, "warmup_runs", 0)
    timed_repetitions = _validate_integer(
        timed_repetitions, "timed_repetitions", 1
    )

    if n_samples <= n_features:
        raise ValueError("'n_samples' must be greater than 'n_features'.")

    experiments = []

    for index, (experiment_id, case_name) in enumerate(CASES):
        case_seed = seed + index

        # Reuse the existing numerical validation for this exact case.
        experiment = _run_case(
            experiment_id=experiment_id,
            case_name=case_name,
            seed=case_seed,
            n_samples=n_samples,
            n_features=n_features,
        )

        # The generator is deterministic: this recreates the same matrix.
        matrix, _ = _generate_case_matrix(
            case_name, n_samples, n_features, case_seed
        )

        for method in METHODS:
            experiment["methods"][method]["timing"] = _benchmark_method(
                matrix,
                method,
                warmup_runs,
                timed_repetitions,
            )

        experiments.append(experiment)

    report = {
        "schema_version": 1,
        "suite_id": "TM-PCA-PERFORMANCE-01",
        "purpose": "pca_performance_and_numerical_validation",
        "performance_measurements": {
            "performed": True,
            "timing_scope": (
                "PCA function call only; matrix generation and numerical "
                "validation are excluded from timed regions."
            ),
            "warmup_runs": warmup_runs,
            "timed_repetitions": timed_repetitions,
        },
        "parameters": {
            "n_samples": n_samples,
            "n_features": n_features,
            "dtype": "float64",
            "base_random_seed": seed,
            "case_seed_policy": "base seed plus case index (0, 1, 2)",
        },
        "environment": collect_environment_metadata(
            project_root=project_root
        ),
        "experiments": experiments,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Rapport : {path}")
    print(f"Matrice : {n_samples} x {n_features}")
    print(f"Échauffements : {warmup_runs}")
    print(f"Répétitions chronométrées : {timed_repetitions}")
    print(f"Expériences : {len(experiments)}")
    print(f"Méthodes : {', '.join(METHODS)}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PCA covariance versus direct SVD."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-repetitions", type=int, default=20)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    run_pca_performance_benchmark(
        output_path=args.output,
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        timed_repetitions=args.timed_repetitions,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()
