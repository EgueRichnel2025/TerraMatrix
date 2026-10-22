import numpy as np
import pytest

from terramatrix.numerical import MatrixDiagnostics, diagnose_matrix


def test_diagnose_identity_matrix() -> None:
    a = np.eye(3)

    result = diagnose_matrix(a)

    assert isinstance(result, MatrixDiagnostics)
    assert result.shape == (3, 3)
    assert result.numerical_rank == 3
    assert result.maximum_rank == 3
    assert not result.is_rank_deficient
    assert result.rank_tolerance == pytest.approx(
        3 * np.finfo(np.float64).eps
    )
    assert result.largest_singular_value == pytest.approx(1.0)
    assert result.smallest_singular_value == pytest.approx(1.0)
    assert result.condition_number_2 == pytest.approx(1.0)


@pytest.mark.parametrize("smallest", [1e-4, 1e-8, 1e-12])
def test_diagnose_well_defined_full_rank_conditioning(
    smallest: float,
) -> None:
    a = np.diag([1.0, smallest])

    result = diagnose_matrix(a)

    assert result.numerical_rank == 2
    assert result.maximum_rank == 2
    assert not result.is_rank_deficient
    assert result.condition_number_2 == pytest.approx(
        1.0 / smallest,
        rel=1e-12,
    )


def test_condition_number_increases_as_smallest_singular_value_decreases() -> None:
    conditions = [
        diagnose_matrix(np.diag([1.0, smallest])).condition_number_2
        for smallest in [1e-4, 1e-8, 1e-12]
    ]

    assert conditions[0] < conditions[1] < conditions[2]


def test_diagnose_tall_full_rank_matrix() -> None:
    a = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )

    result = diagnose_matrix(a)

    assert result.shape == (4, 2)
    assert result.maximum_rank == 2
    assert result.numerical_rank == 2
    assert not result.is_rank_deficient
    assert np.isfinite(result.condition_number_2)


def test_diagnose_wide_full_row_rank_matrix() -> None:
    a = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    result = diagnose_matrix(a)

    assert result.shape == (2, 3)
    assert result.maximum_rank == 2
    assert result.numerical_rank == 2
    assert not result.is_rank_deficient
    assert result.condition_number_2 == pytest.approx(2.0)


def test_diagnose_rank_deficient_matrix() -> None:
    a = np.array(
        [
            [1.0, 2.0, 1.0],
            [3.0, 6.0, 3.0],
            [2.0, 4.0, 2.0],
            [5.0, 10.0, 5.0],
        ]
    )

    result = diagnose_matrix(a)

    assert result.numerical_rank == 1
    assert result.maximum_rank == 3
    assert result.is_rank_deficient
    assert np.isinf(result.condition_number_2)


def test_diagnose_zero_matrix() -> None:
    result = diagnose_matrix(np.zeros((3, 2)))

    assert result.numerical_rank == 0
    assert result.maximum_rank == 2
    assert result.largest_singular_value == 0.0
    assert result.smallest_singular_value == 0.0
    assert result.is_rank_deficient
    assert np.isinf(result.condition_number_2)


def test_custom_rank_tolerance_changes_numerical_rank() -> None:
    a = np.diag([1.0, 1e-6])

    strict = diagnose_matrix(a, rank_tolerance=1e-8)
    loose = diagnose_matrix(a, rank_tolerance=1e-5)

    assert strict.numerical_rank == 2
    assert not strict.is_rank_deficient
    assert strict.condition_number_2 == pytest.approx(1e6)

    assert loose.numerical_rank == 1
    assert loose.is_rank_deficient
    assert loose.rank_tolerance == pytest.approx(1e-5)
    assert np.isinf(loose.condition_number_2)


@pytest.mark.parametrize(
    "tolerance",
    [-1.0, np.nan, np.inf, True, "1e-6", 1.0 + 1.0j],
)
def test_diagnose_rejects_invalid_rank_tolerance(
    tolerance: object,
) -> None:
    with pytest.raises(ValueError, match="rank_tolerance"):
        diagnose_matrix(np.eye(2), rank_tolerance=tolerance)


def test_diagnose_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        diagnose_matrix(np.array([1.0, 2.0]))


def test_diagnose_rejects_non_finite_matrix() -> None:
    a = np.array([[1.0, np.nan], [0.0, 2.0]])

    with pytest.raises(ValueError, match="finite values"):
        diagnose_matrix(a)


def test_diagnose_rejects_complex_matrix() -> None:
    a = np.array([[1.0 + 1.0j, 0.0], [0.0, 2.0]])

    with pytest.raises(ValueError, match="real values"):
        diagnose_matrix(a)
