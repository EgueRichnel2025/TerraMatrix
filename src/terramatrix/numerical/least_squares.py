"""Least-squares solvers based on BLAS and LAPACK routines."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular

from terramatrix.numerical.blas import (
    _as_finite_matrix,
    _as_finite_vector,
    dsyrk,
)
from terramatrix.numerical.lapack import (
    qr_factorization,
    svd_factorization,
)


def _prepare_problem(
    a: ArrayLike,
    b: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate an overdetermined or square least-squares problem."""
    matrix = _as_finite_matrix(a, "a")
    response = _as_finite_vector(b, "b")

    if matrix.shape[0] != response.size:
        raise ValueError(
            "The number of rows in 'a' must equal the length of 'b'."
        )

    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError(
            "'a' must have at least as many rows as columns."
        )

    return matrix, response


def _validate_rank_tolerance(value: Real) -> float:
    """Validate an absolute singular-value or diagonal cutoff."""
    message = (
        "'rank_tolerance' must be a finite, non-negative real scalar."
    )

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)

    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(message)

    return tolerance


def _resolve_rank_tolerance(
    rank_tolerance: Real | None,
    scale: float,
    shape: tuple[int, int],
) -> float:
    """Return an absolute cutoff using the provided scale."""
    if rank_tolerance is not None:
        return _validate_rank_tolerance(rank_tolerance)

    return float(
        max(shape) * np.finfo(np.float64).eps * scale
    )


def _validate_solution(
    coefficients: NDArray[np.float64],
    method: str,
) -> NDArray[np.float64]:
    """Reject a solver result containing non-finite coefficients."""
    if not np.isfinite(coefficients).all():
        raise FloatingPointError(
            f"{method} produced non-finite coefficients."
        )

    return np.asarray(coefficients, dtype=np.float64)


def solve_normal_equations(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.float64]:
    """Solve least squares using the normal equations.

    The Gram matrix is formed with BLAS DSYRK. This method is a
    comparison baseline; forming A.T @ A squares the 2-norm condition
    number in exact arithmetic when A has full column rank.
    """
    matrix, response = _prepare_problem(a, b)

    gram = dsyrk(matrix, trans="T")
    right_hand_side = matrix.T @ response

    coefficients = np.linalg.solve(gram, right_hand_side)

    return _validate_solution(coefficients, "Normal equations")


def solve_qr(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.float64]:
    """Solve least squares using the existing unpivoted QR factorization.

    This method does not reveal numerical rank. It may fail or return
    inaccurate coefficients for rank-deficient or ill-conditioned inputs.
    """
    matrix, response = _prepare_problem(a, b)

    q, r = qr_factorization(matrix)
    projected_response = q.T @ response

    coefficients = solve_triangular(
        r,
        projected_response,
        lower=False,
        check_finite=False,
    )

    return _validate_solution(coefficients, "Unpivoted QR")


def solve_pivoted_qr(
    a: ArrayLike,
    b: ArrayLike,
    *,
    rank_tolerance: Real | None = None,
) -> NDArray[np.float64]:
    """Solve least squares using column-pivoted QR.

    The absolute cutoff is applied to the diagonal of R. By default,
    it is scaled using the magnitude of the first diagonal entry.
    In a rank-deficient problem, free permuted coefficients are set
    to zero. This produces a basic solution, not necessarily the
    minimum-norm solution.
    """
    matrix, response = _prepare_problem(a, b)

    q, r, pivots = scipy_qr(
        matrix,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )

    diagonal = np.abs(np.diag(r))
    scale = float(diagonal[0]) if diagonal.size else 0.0

    tolerance = _resolve_rank_tolerance(
        rank_tolerance,
        scale,
        matrix.shape,
    )

    # Count consecutive diagonal entries above the cutoff.
    # Column-pivoted QR is rank-revealing, but its diagonal is still
    # an estimate of rank rather than an exact singular-value spectrum.
    rank = 0
    for value in diagonal:
        if value <= tolerance:
            break
        rank += 1

    permuted_coefficients = np.zeros(
        matrix.shape[1],
        dtype=np.float64,
    )

    if rank > 0:
        leading_r = r[:rank, :rank]
        projected_response = q[:, :rank].T @ response

        permuted_coefficients[:rank] = solve_triangular(
            leading_r,
            projected_response,
            lower=False,
            check_finite=False,
        )

    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    coefficients[pivots] = permuted_coefficients

    return _validate_solution(coefficients, "Pivoted QR")


def solve_svd(
    a: ArrayLike,
    b: ArrayLike,
    *,
    rank_tolerance: Real | None = None,
) -> NDArray[np.float64]:
    """Solve least squares using a truncated SVD pseudoinverse.

    Singular values strictly greater than the absolute cutoff are
    retained. The default cutoff is

        max(m, n) * eps(float64) * largest_singular_value.

    The result is the minimum-Euclidean-norm least-squares solution
    for the retained singular subspace.
    """
    matrix, response = _prepare_problem(a, b)

    u, singular_values, vt = svd_factorization(matrix)

    largest_singular_value = (
        float(singular_values[0]) if singular_values.size else 0.0
    )

    tolerance = _resolve_rank_tolerance(
        rank_tolerance,
        largest_singular_value,
        matrix.shape,
    )

    retained = singular_values > tolerance
    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)

    if np.any(retained):
        projected_response = u[:, retained].T @ response
        scaled_response = projected_response / singular_values[retained]
        coefficients = vt[retained, :].T @ scaled_response

    return _validate_solution(coefficients, "SVD")
