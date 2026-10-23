"""Tests for reproducible least-squares comparison experiments."""

import json
from pathlib import Path

import numpy as np
import pytest

from terramatrix.experiments.least_squares_comparison import (
    _generate_case_matrix,
    run_least_squares_comparison_experiments,
)


def test_well_conditioned_matrix_has_prescribed_spectrum() -> None:
    rng = np.random.default_rng(20261022)

    matrix, generation = _generate_case_matrix(
        "well_conditioned",
        n_samples=30,
        n_features=5,
        rng=rng,
    )

    singular_values = np.linalg.svd(matrix, compute_uv=False)

    np.testing.assert_allclose(
        singular_values,
        generation["prescribed_singular_values"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.linalg.cond(matrix) == pytest.approx(2.0, rel=1e-11)
    assert generation["prescribed_condition_number_2"] == 2.0


@pytest.mark.parametrize("target", [1e2, 1e4, 1e6])
def test_ill_conditioned_matrix_has_target_condition_number(
    target: float,
) -> None:
    rng = np.random.default_rng(20261023)

    matrix, generation = _generate_case_matrix(
        "ill_conditioned",
        n_samples=30,
        n_features=5,
        rng=rng,
        target_condition_number=target,
    )

    assert generation["prescribed_condition_number_2"] == target
    assert np.linalg.cond(matrix) == pytest.approx(target, rel=1e-8)


def test_rank_deficient_matrix_has_exact_duplicate_columns() -> None:
    rng = np.random.default_rng(20261024)

    matrix, generation = _generate_case_matrix(
        "rank_deficient",
        n_samples=30,
        n_features=5,
        rng=rng,
    )

    assert generation["collinearity_epsilon"] == 0.0
    np.testing.assert_array_equal(matrix[:, 0], matrix[:, -1])
    assert np.linalg.matrix_rank(matrix) == 4


def test_runner_writes_complete_json_report(tmp_path: Path) -> None:
    destination = tmp_path / "reports" / "least-squares.json"

    report = run_least_squares_comparison_experiments(
        output_path=destination,
        n_samples=30,
        n_features=5,
        seed=20261022,
        target_condition_numbers=(1e2, 1e4, 1e6),
        warmup_runs=0,
        timed_repetitions=2,
        project_root=tmp_path,
    )

    assert destination.is_file()
    content = destination.read_text(encoding="utf-8")
    assert content.endswith("\n")

    stored_report = json.loads(content)
    assert stored_report == report
    assert report["schema_version"] == 1
    assert report["suite_id"] == "TM-LS-COMPARISON-01"
    assert report["performance_measurements"]["performed"] is True

    experiments = report["experiments"]
    assert len(experiments) == 5

    assert experiments[0]["experiment_id"] == "TM-LS-01"
    assert experiments[-1]["experiment_id"] == "TM-LS-03"

    ill_conditioned = experiments[1:4]
    assert [
        case["rank_diagnostics"]["target_condition_number_2"]
        for case in ill_conditioned
    ] == [1e2, 1e4, 1e6]

    assert [
        case["random_seed"] for case in experiments
    ] == [20261022] * 5

    ill_conditioned_family = experiments[1:4]

    assert all(
        case["reference_coefficients"]
        == ill_conditioned_family[0]["reference_coefficients"]
        for case in ill_conditioned_family
    )
    assert all(
        case["noise"] == ill_conditioned_family[0]["noise"]
        for case in ill_conditioned_family
    )
    assert all(
        case["reference_coefficients_sha256"]
        == ill_conditioned_family[0]["reference_coefficients_sha256"]
        for case in ill_conditioned_family
    )
    assert len(
        {case["matrix"]["sha256"] for case in ill_conditioned_family}
    ) == len(ill_conditioned_family)
    assert len(
        {case["response_sha256"] for case in ill_conditioned_family}
    ) == len(ill_conditioned_family)

    for experiment in experiments:
        diagnostics = experiment["rank_diagnostics"]
        methods = experiment["methods"]

        expected_rank = 4 if experiment["case"] == "rank_deficient" else 5
        assert diagnostics["numerical_rank"] == expected_rank
        assert diagnostics["expectation_matches"] is True

        assert set(methods) == {
            "normal_equations",
            "qr_unpivoted",
            "qr_pivoted",
            "svd_pseudoinverse",
        }

        for method in methods.values():
            assert method["solver_status"] in {
                "success",
                "partial_failure",
                "failure",
            }

            timing = method["timing"]
            assert timing["performed"] is True
            assert timing["warmup_runs"] == 0
            assert timing["timed_repetitions"] == 2
            assert len(timing["raw_execution_times_seconds"]) == 2
            assert len(timing["timed_trials"]) == 2

            assert all(
                np.isfinite(value) and value >= 0.0
                for value in timing["raw_execution_times_seconds"]
            )

            if method["solver_status"] == "success":
                assert timing["median_seconds"] is not None
                assert np.isfinite(timing["median_seconds"])
                assert timing["interquartile_range_seconds"] is not None
            else:
                assert timing["median_seconds"] is None
                assert (
                    timing["status"] == "incomplete_due_to_solver_failure"
                )

        # Coefficient recovery is not used as the correctness criterion
        # for an exactly rank-deficient design matrix.
        if experiment["case"] == "rank_deficient":
            for method in methods.values():
                if method["metrics"] is not None:
                    assert (
                        method["metrics"]["relative_coefficient_error_2"]
                        is None
                    )


def test_runner_is_reproducible_for_fixed_seed(tmp_path: Path) -> None:
    common_parameters = {
        "n_samples": 20,
        "n_features": 4,
        "seed": 71,
        "noise_std": 1e-3,
        "target_condition_numbers": (1e2,),
        "warmup_runs": 0,
        "timed_repetitions": 1,
        "project_root": tmp_path,
    }

    first = run_least_squares_comparison_experiments(
        output_path=tmp_path / "first.json",
        **common_parameters,
    )
    second = run_least_squares_comparison_experiments(
        output_path=tmp_path / "second.json",
        **common_parameters,
    )

    for first_case, second_case in zip(
        first["experiments"],
        second["experiments"],
        strict=True,
    ):
        assert first_case["experiment_id"] == second_case["experiment_id"]
        assert first_case["random_seed"] == second_case["random_seed"]
        assert first_case["matrix_generation"] == second_case["matrix_generation"]
        assert first_case["noise"] == second_case["noise"]
        assert first_case["reference_coefficients"] == (
            second_case["reference_coefficients"]
        )
        assert first_case["rank_diagnostics"] == second_case["rank_diagnostics"]

        for method_name in first_case["methods"]:
            first_metrics = first_case["methods"][method_name]["metrics"]
            second_metrics = second_case["methods"][method_name]["metrics"]

            if first_metrics is None or second_metrics is None:
                assert first_metrics is second_metrics
            else:
                np.testing.assert_allclose(
                    first_metrics["coefficients"],
                    second_metrics["coefficients"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                assert first_metrics["residual_norm_2"] == pytest.approx(
                    second_metrics["residual_norm_2"],
                    rel=1e-12,
                    abs=1e-12,
                )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (
            {"n_samples": 20, "n_features": 20},
            "greater than",
        ),
        (
            {"n_samples": 20, "n_features": 4, "seed": -1},
            "seed",
        ),
        (
            {"n_samples": 20, "n_features": 4, "warmup_runs": -1},
            "warmup_runs",
        ),
        (
            {"n_samples": 20, "n_features": 4, "timed_repetitions": 0},
            "timed_repetitions",
        ),
        (
            {
                "n_samples": 20,
                "n_features": 4,
                "target_condition_numbers": (1.0, 0.5),
            },
            "target_condition_numbers entry",
        ),
        (
            {
                "n_samples": 20,
                "n_features": 4,
                "target_condition_numbers": (),
            },
            "must not be empty",
        ),
    ],
)
def test_runner_rejects_invalid_parameters(
    tmp_path: Path,
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_least_squares_comparison_experiments(
            output_path=tmp_path / "invalid.json",
            project_root=tmp_path,
            **parameters,
        )


def test_runner_records_partial_solver_failure_and_warning() -> None:
    import warnings

    from terramatrix.experiments.least_squares_comparison import _run_method
    from terramatrix.numerical import diagnose_matrix

    matrix = np.eye(3)
    response = np.array([1.0, 2.0, 3.0])
    beta_true = response.copy()
    calls = 0

    def flaky_solver(a, b):
        nonlocal calls
        calls += 1

        if calls == 2:
            warnings.warn(
                "Controlled warning for testing.",
                RuntimeWarning,
                stacklevel=2,
            )
            raise np.linalg.LinAlgError("Controlled solver failure.")

        return np.linalg.lstsq(a, b, rcond=None)[0]

    result = _run_method(
        method_name="normal_equations",
        solver=flaky_solver,
        matrix=matrix,
        response=response,
        beta_true=beta_true,
        coefficient_error_meaningful=True,
        matrix_diagnostics=diagnose_matrix(matrix),
        warmup_runs=0,
        timed_repetitions=2,
    )

    assert result["solver_status"] == "partial_failure"
    assert result["metrics"] is not None

    timing = result["timing"]
    assert timing["status"] == "incomplete_due_to_solver_failure"
    assert timing["median_seconds"] is None
    assert timing["interquartile_range_seconds"] is None
    assert len(timing["raw_execution_times_seconds"]) == 2

    trials = timing["timed_trials"]
    assert [trial["status"] for trial in trials] == [
        "success",
        "failure",
    ]
    assert trials[1]["error_type"] == "LinAlgError"
    assert trials[1]["error_message"] == "Controlled solver failure."

    assert {
        "category": "RuntimeWarning",
        "message": "Controlled warning for testing.",
    } in result["warnings"]
