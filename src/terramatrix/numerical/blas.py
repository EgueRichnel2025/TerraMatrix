"""Validated BLAS operations for TerraMatrix."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg.blas import get_blas_funcs


def _as_finite_matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert a real, finite, non-empty two-dimensional input to float64."""
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be a valid numeric matrix.") from exc

    if np.iscomplexobj(matrix):
        raise ValueError(f"'{name}' must contain real values, not complex values.")

    try:
        matrix = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{name}' must be convertible to float64.") from exc

    if matrix.ndim != 2:
        raise ValueError(f"'{name}' must be a two-dimensional matrix.")

    if 0 in matrix.shape:
        raise ValueError(f"'{name}' must not have an empty dimension.")

    if not np.isfinite(matrix).all():
        raise ValueError(f"'{name}' must contain only finite values.")

    return np.asfortranarray(matrix)


def _as_finite_scalar(value: Real, name: str) -> float:
    """Validate and convert a real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"'{name}' must be a real scalar.")

    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{name}' must be a finite real scalar.") from exc

    if not np.isfinite(scalar):
        raise ValueError(f"'{name}' must be finite.")

    return scalar


def dgemm(
    a: ArrayLike,
    b: ArrayLike,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    c: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute C_result = alpha * A @ B + beta * C_initial using BLAS DGEMM.

    Inputs are converted to finite float64 matrices in Fortran order.

    If C_initial is omitted, beta must be zero. Otherwise, C_initial
    must have shape (A.shape[0], B.shape[1]).

    The function validates inputs and prepares their memory layout before
    calling the low-level BLAS routine. Its total execution time therefore
    includes validation and preparation overhead.
    """
    a_matrix = _as_finite_matrix(a, "a")
    b_matrix = _as_finite_matrix(b, "b")

    if a_matrix.shape[1] != b_matrix.shape[0]:
        raise ValueError(
            "Incompatible matrix dimensions: "
            f"a has shape {a_matrix.shape}, "
            f"b has shape {b_matrix.shape}."
        )

    alpha_value = _as_finite_scalar(alpha, "alpha")
    beta_value = _as_finite_scalar(beta, "beta")

    output_shape = (a_matrix.shape[0], b_matrix.shape[1])

    if c is None:
        if beta_value != 0.0:
            raise ValueError("'c' must be provided when 'beta' is non-zero.")

        c_initial = np.zeros(output_shape, dtype=np.float64, order="F")
    else:
        c_initial = _as_finite_matrix(c, "c")

        if c_initial.shape != output_shape:
            raise ValueError(
                f"'c' must have shape {output_shape}, "
                f"received {c_initial.shape}."
            )

        # BLAS may overwrite its output argument. Preserve the caller's input.
        c_initial = np.array(c_initial, dtype=np.float64, order="F", copy=True)

    gemm = get_blas_funcs(
        "gemm",
        arrays=(a_matrix, b_matrix),
        ilp64="preferred",
    )

    result = gemm(
        alpha_value,
        a_matrix,
        b_matrix,
        beta=beta_value,
        c=c_initial,
        overwrite_c=1,
    )

    if not np.isfinite(result).all():
        raise FloatingPointError(
            "DGEMM produced non-finite output; check input magnitudes."
        )

    return np.asarray(result, dtype=np.float64)
