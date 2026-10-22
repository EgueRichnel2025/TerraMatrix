"""Rank and condition-number diagnostics for TerraMatrix matrices."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from terramatrix.numerical.lapack import svd_factorization


@dataclass(frozen=True, slots=True)
class MatrixDiagnostics:
    """Numerical rank and conditioning information for a matrix."""

    shape: tuple[int, int]
    numerical_rank: int
    maximum_rank: int
    rank_tolerance: float
    largest_singular_value: float
    smallest_singular_value: float
    condition_number_2: float

    @property
    def is_rank_deficient(self) -> bool:
        """Return whether the estimated rank is below the maximum possible rank."""
        return self.numerical_rank < self.maximum_rank


def _validate_rank_tolerance(value: Real) -> float:
    """Validate an absolute singular-value threshold."""
    error_message = (
        "'rank_tolerance' must be a finite, non-negative real scalar."
    )

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(error_message)

    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error_message) from exc

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(error_message)

    return tolerance


def diagnose_matrix(
    a: np.ndarray,
    *,
    rank_tolerance: Real | None = None,
) -> MatrixDiagnostics:
    """Estimate a matrix's numerical rank and its 2-norm condition number.

    The matrix is decomposed using the existing LAPACK-backed SVD operation.

    If rank_tolerance is None, the default absolute threshold is

        max(m, n) * eps(float64) * largest_singular_value.

    Otherwise, rank_tolerance is interpreted as an absolute threshold.
    A singular value contributes to the numerical rank only when it is
    strictly greater than this threshold.

    The maximum possible rank is min(m, n). If the estimated numerical rank
    is smaller, condition_number_2 is reported as infinity. Otherwise, it is
    computed as the largest singular value divided by the smallest one.

    Numerical rank depends on the selected threshold and should not be
    interpreted as a proof of exact mathematical rank.
    """
    u, singular_values, vt = svd_factorization(a)

    shape = (int(u.shape[0]), int(vt.shape[1]))
    maximum_rank = int(singular_values.size)

    largest_singular_value = float(singular_values[0])
    smallest_singular_value = float(singular_values[-1])

    if rank_tolerance is None:
        tolerance = (
            max(shape)
            * np.finfo(np.float64).eps
            * largest_singular_value
        )
    else:
        tolerance = _validate_rank_tolerance(rank_tolerance)

    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    is_rank_deficient = numerical_rank < maximum_rank

    if is_rank_deficient:
        condition_number_2 = float("inf")
    else:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            condition_number_2 = float(
                np.divide(
                    np.float64(largest_singular_value),
                    np.float64(smallest_singular_value),
                )
            )

    return MatrixDiagnostics(
        shape=shape,
        numerical_rank=numerical_rank,
        maximum_rank=maximum_rank,
        rank_tolerance=float(tolerance),
        largest_singular_value=largest_singular_value,
        smallest_singular_value=smallest_singular_value,
        condition_number_2=condition_number_2,
    )
