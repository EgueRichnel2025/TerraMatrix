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


def dsyrk(
    a: ArrayLike,
    *,
    trans: str = "T",
    alpha: float = 1.0,
    beta: float = 0.0,
    c: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute a symmetric rank-k update using BLAS DSYRK.

    With trans='T', compute:

        result = alpha * A.T @ A + beta * C

    With trans='N', compute:

        result = alpha * A @ A.T + beta * C

    The input matrix and any initial C matrix must contain finite,
    real float64-compatible values. The initial C matrix must be
    symmetric within a numerical tolerance.

    If C is omitted, beta must be zero. The caller's input arrays
    are not modified.

    The returned matrix is explicitly reconstructed from the triangle
    computed by BLAS, so the complete result is symmetric.

    Validation and memory-layout preparation are included in the
    execution time of this wrapper.
    """
    if trans not in ("N", "T"):
        raise ValueError("'trans' must be either 'N' or 'T'.")

    a_matrix = _as_finite_matrix(a, "a")
    alpha_value = _as_finite_scalar(alpha, "alpha")
    beta_value = _as_finite_scalar(beta, "beta")

    if trans == "T":
        output_dimension = a_matrix.shape[1]
    else:
        output_dimension = a_matrix.shape[0]

    output_shape = (output_dimension, output_dimension)

    if c is None:
        if beta_value != 0.0:
            raise ValueError("'c' must be provided when 'beta' is non-zero.")

        c_initial = np.zeros(
            output_shape,
            dtype=np.float64,
            order="F",
        )
    else:
        c_initial = _as_finite_matrix(c, "c")

        if c_initial.shape != output_shape:
            raise ValueError(
                f"'c' must have shape {output_shape}, "
                f"received {c_initial.shape}."
            )

        if not np.allclose(
            c_initial,
            c_initial.T,
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError("'c' must be symmetric.")

        # Use an independent, explicitly symmetric copy.
        c_initial = np.asfortranarray(
            0.5 * c_initial + 0.5 * c_initial.T
        )

    syrk = get_blas_funcs(
        "syrk",
        arrays=(a_matrix,),
        ilp64="preferred",
    )

    result = syrk(
        alpha_value,
        a_matrix,
        beta=beta_value,
        c=c_initial,
        trans=1 if trans == "T" else 0,
        lower=1,
        overwrite_c=1,
    )

    if not np.isfinite(result).all():
        raise FloatingPointError(
            "DSYRK produced non-finite output; check input magnitudes."
        )

    # BLAS DSYRK computes only the selected triangle.
    # Reconstruct the other triangle explicitly.
    lower_triangle = np.tril(result)
    symmetric_result = lower_triangle + np.tril(result, k=-1).T

    return np.asarray(symmetric_result, dtype=np.float64)


def _as_finite_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert a real, finite, non-empty one-dimensional input to float64."""
    try:
        vector = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be a valid numeric vector.") from exc

    if np.iscomplexobj(vector):
        raise ValueError(f"'{name}' must contain real values, not complex values.")

    try:
        vector = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{name}' must be convertible to float64.") from exc

    if vector.ndim != 1:
        raise ValueError(f"'{name}' must be a one-dimensional vector.")

    if vector.size == 0:
        raise ValueError(f"'{name}' must not be empty.")

    if not np.isfinite(vector).all():
        raise ValueError(f"'{name}' must contain only finite values.")

    return np.ascontiguousarray(vector)


def dgemv(
    a: ArrayLike,
    x: ArrayLike,
    *,
    trans: str = "N",
    alpha: float = 1.0,
    beta: float = 0.0,
    y: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute a matrix-vector product using BLAS DGEMV.

    With trans='N', compute:

        result = alpha * A @ x + beta * y_initial

    With trans='T', compute:

        result = alpha * A.T @ x + beta * y_initial

    Inputs must be finite, real, and float64-compatible. The input matrix
    must be two-dimensional, and x and any supplied y must be vectors.

    If y is omitted, beta must be zero. The caller's input arrays are
    not modified.

    Input validation and memory-layout preparation are included in the
    execution time of this wrapper.
    """
    if trans not in ("N", "T"):
        raise ValueError("'trans' must be either 'N' or 'T'.")

    a_matrix = _as_finite_matrix(a, "a")
    x_vector = _as_finite_vector(x, "x")
    alpha_value = _as_finite_scalar(alpha, "alpha")
    beta_value = _as_finite_scalar(beta, "beta")

    if trans == "N":
        input_dimension = a_matrix.shape[1]
        output_dimension = a_matrix.shape[0]
        transpose_flag = 0
    else:
        input_dimension = a_matrix.shape[0]
        output_dimension = a_matrix.shape[1]
        transpose_flag = 1

    if x_vector.size != input_dimension:
        raise ValueError(
            f"'x' must have length {input_dimension} for trans='{trans}', "
            f"received {x_vector.size}."
        )

    if y is None:
        if beta_value != 0.0:
            raise ValueError("'y' must be provided when 'beta' is non-zero.")

        y_initial = np.zeros(output_dimension, dtype=np.float64)
    else:
        y_initial = _as_finite_vector(y, "y")

        if y_initial.size != output_dimension:
            raise ValueError(
                f"'y' must have length {output_dimension}, "
                f"received {y_initial.size}."
            )

        # BLAS may overwrite its output argument. Preserve the caller's input.
        y_initial = np.array(y_initial, dtype=np.float64, copy=True)

    gemv = get_blas_funcs(
        "gemv",
        arrays=(a_matrix, x_vector),
        ilp64="preferred",
    )

    result = gemv(
        alpha_value,
        a_matrix,
        x_vector,
        beta=beta_value,
        y=y_initial,
        trans=transpose_flag,
        overwrite_y=1,
    )

    if not np.isfinite(result).all():
        raise FloatingPointError(
            "DGEMV produced non-finite output; check input magnitudes."
        )

    return np.asarray(result, dtype=np.float64)
