import json
from pathlib import Path

import numpy as np
import pytest

from terramatrix.experiments.pca_comparison import (
    _generate_case_matrix,
    run_pca_comparison_experiments,
)


def test_well_conditioned_matrix_has_prescribed_spectrum() -> None:
    matrix, generation = _generate_case_matrix(
        "well_conditioned",
        n_samples=20,
        n_features=4,
        seed=123,
    )

    singular_values = np.linalg.svd(matrix, compute_uv=False)

    np.testing.assert_allclose(
        singular_values,
        generation["prescribed_singular_values"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.linalg.cond(matrix) == pytest.approx(2.0, rel=1e-11)


def test_ill_conditioned_matrix_has_prescribed_conditioning() -> None:
    matrix, generation = _generate_case_matrix(
        "ill_conditioned",
        n_samples=20,
        n_features=4,
        seed=124,
    )

    assert generation["prescribed_condition_number_2"] == pytest.approx(1e6)
    assert np.linalg.cond(matrix) == pytest.approx(1e6, rel=1e-8)


def test_rank_deficient_matrix_has_exact_duplicate_columns() -> None:
    matrix, generation = _generate_case_matrix(
        "rank_deficient",
        n_samples=20,
        n_features=4,
        seed=125,
    )

    assert generation["collinearity_epsilon"] == 0.0
    np.testing.assert_array_equal(matrix[:, 0], matrix[:, -1])


def test_experiment_runner_writes_reproducible_json_report(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "reports" / "pca-comparison.json"

    report = run_pca_comparison_experiments(
        output_path=destination,
        n_samples=20,
        n_features=4,
        seed=20261022,
        project_root=tmp_path,
    )

    assert destination.is_file()

    content = destination.read_text(encoding="utf-8")
    assert content.endswith("\n")

    stored_report = json.loads(content)
    assert stored_report == report
    assert report["schema_version"] == 1
    assert report["suite_id"] == "TM-PCA-COMPARISON-01"
    assert report["performance_measurements"]["performed"] is False

    assert [case["experiment_id"] for case in report["experiments"]] == [
        "TM-PCA-01",
        "TM-PCA-02",
        "TM-PCA-03",
    ]

    assert [case["random_seed"] for case in report["experiments"]] == [
        20261022,
        20261023,
        20261024,
    ]

    assert report["environment"]["project"]["name"] == "terramatrix"


def test_report_records_numerical_rank_and_conditioning(
    tmp_path: Path,
) -> None:
    report = run_pca_comparison_experiments(
        output_path=tmp_path / "report.json",
        n_samples=20,
        n_features=4,
        seed=7,
        project_root=tmp_path,
    )

    well_conditioned, ill_conditioned, rank_deficient = report["experiments"]

    assert well_conditioned["rank_diagnostics"]["numerical_rank"] == 4
    assert ill_conditioned["rank_diagnostics"]["numerical_rank"] == 4
    assert rank_deficient["rank_diagnostics"]["numerical_rank"] == 3

    assert (
        well_conditioned["rank_diagnostics"]["measured_condition_number_2"]
        == pytest.approx(2.0, rel=1e-10)
    )
    assert (
        ill_conditioned["rank_diagnostics"]["measured_condition_number_2"]
        == pytest.approx(1e6, rel=1e-8)
    )

    assert (
        rank_deficient["rank_diagnostics"]["measured_condition_number_2"]
        is None
    )
    assert (
        rank_deficient["rank_diagnostics"]["condition_number_status"]
        == "infinite_due_to_numerical_rank_deficiency"
    )


def test_both_pca_methods_are_reported_for_every_case(
    tmp_path: Path,
) -> None:
    report = run_pca_comparison_experiments(
        output_path=tmp_path / "report.json",
        n_samples=20,
        n_features=4,
        seed=42,
        project_root=tmp_path,
    )

    for experiment in report["experiments"]:
        methods = experiment["methods"]

        assert set(methods) == {"covariance", "svd"}

        for metrics in methods.values():
            assert metrics["n_components"] == 4
            assert len(metrics["explained_variance"]) == 4
            assert len(metrics["explained_variance_ratio"]) == 4
            assert np.all(np.asarray(metrics["explained_variance"]) >= 0.0)
            assert np.isfinite(
                metrics["reconstruction_relative_error_fro"]
            )
            assert np.isfinite(
                metrics["component_orthogonality_error_fro"]
            )

        for metric in experiment["comparison"].values():
            assert np.isfinite(metric)
            assert metric >= 0.0


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"n_samples": 20, "n_features": 20}, "greater than"),
        ({"n_samples": 20, "n_features": 1}, "n_features"),
        ({"n_samples": 20, "n_features": 4, "seed": -1}, "seed"),
        ({"n_samples": True, "n_features": 4}, "n_samples"),
        ({"n_samples": 20, "n_features": 4, "seed": 1.5}, "seed"),
    ],
)
def test_experiment_runner_rejects_invalid_parameters(
    tmp_path: Path,
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_pca_comparison_experiments(
            output_path=tmp_path / "report.json",
            project_root=tmp_path,
            **parameters,
        )
