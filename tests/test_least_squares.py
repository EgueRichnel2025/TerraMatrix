"""Tests for TerraMatrix least-squares solvers."""

import numpy as np
import pytest

from terramatrix.numerical.least_squares import (
    solve_normal_equations,
    solve_pivoted_qr,
    solve_qr,
    solve_svd,
)

SOLVERS = [
    solve_normal_equations,
    solve_qr,
    solve_pivoted_qr,
    solve_svd,
]


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_match_reference_on_well_conditioned_problem(solver) -> None:
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal((80, 6))
    beta_true = rng.standard_normal(6)
    response = a @ beta_true + rng.normal(scale=1e-4, size=80)

    expected, _, _, _ = np.linalg.lstsq(a, response, rcond=None)
    actual = solver(a, response)

    assert actual.shape == (6,)
    assert actual.dtype == np.float64
    assert np.isfinite(actual).all()

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_svd_returns_minimum_norm_solution_for_duplicate_columns() -> None:
    rng = np.random.default_rng(20261023)
    a = rng.standard_normal((30, 3))
    a[:, 2] = a[:, 0]
    response = rng.standard_normal(30)

    actual = solve_svd(a, response)
    expected = np.linalg.pinv(a) @ response

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )

    # The minimum-norm solution gives equal coefficients to identical columns.
    assert actual[0] == pytest.approx(actual[2], rel=1e-10, abs=1e-10)


def test_pivoted_qr_solves_rank_deficient_least_squares_problem() -> None:
    rng = np.random.default_rng(20261024)
    a = rng.standard_normal((50, 4))
    a[:, 3] = a[:, 0]
    response = rng.standard_normal(50)

    actual = solve_pivoted_qr(a, response)
    expected, _, _, _ = np.linalg.lstsq(a, response, rcond=None)

    actual_residual = np.linalg.norm(a @ actual - response)
    expected_residual = np.linalg.norm(a @ expected - response)

    assert np.isfinite(actual).all()
    assert actual_residual == pytest.approx(
        expected_residual,
        rel=1e-10,
        abs=1e-10,
    )


def test_svd_respects_explicit_absolute_rank_tolerance() -> None:
    a = np.array(
        [
            [1.0, 0.0],
            [0.0, 1e-4],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    response = np.array([2.0, 3e-4, 0.0, 0.0])

    actual = solve_svd(a, response, rank_tolerance=1e-3)

    np.testing.assert_allclose(
        actual,
        np.array([2.0, 0.0]),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_reject_underdetermined_problem(solver) -> None:
    a = np.ones((3, 4))
    response = np.ones(3)

    with pytest.raises(ValueError, match="at least as many rows"):
        solver(a, response)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_reject_response_with_wrong_length(solver) -> None:
    a = np.ones((5, 3))
    response = np.ones(4)

    with pytest.raises(ValueError, match="number of rows"):
        solver(a, response)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_reject_non_finite_matrix(solver) -> None:
    a = np.eye(3)
    a[0, 0] = np.nan
    response = np.ones(3)

    with pytest.raises(ValueError, match="finite values"):
        solver(a, response)


@pytest.mark.parametrize("solver", [solve_svd, solve_pivoted_qr])
@pytest.mark.parametrize("bad_tolerance", [-1.0, np.inf, True])
def test_rank_revealing_solvers_reject_invalid_tolerance(
    solver,
    bad_tolerance,
) -> None:
    a = np.eye(3)
    response = np.ones(3)

    with pytest.raises(ValueError, match="rank_tolerance"):
        solver(a, response, rank_tolerance=bad_tolerance)
