"""Validated LAPACK operations for TerraMatrix."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg.lapack import get_lapack_funcs

from terramatrix.numerical.blas import _as_finite_matrix


def qr_factorization(
    a: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute an economic QR factorization using LAPACK GEQRF and ORGQR.

    For a matrix A of shape (m, n), return Q with shape (m, k)
    and R with shape (k, n), where k = min(m, n).

    The factorization satisfies A approximately equal to Q @ R, and
    Q.T @ Q approximately equal to the identity matrix.

    This implementation does not use column pivoting and does not
    estimate numerical rank. Input validation is included.
    """
    matrix = _as_finite_matrix(a, "a")
    m, n = matrix.shape
    k = min(m, n)

    geqrf, orgqr = get_lapack_funcs(
        ("geqrf", "orgqr"),
        arrays=(matrix,),
        ilp64="preferred",
    )

    packed, tau, _, info = geqrf(matrix, overwrite_a=0)

    if info < 0:
        raise ValueError(f"LAPACK GEQRF rejected argument {-info}.")
    if info > 0:
        raise np.linalg.LinAlgError(
            f"LAPACK GEQRF failed with INFO={info}."
        )

    r = np.triu(packed[:k, :])
    q_input = np.array(packed[:, :k], dtype=np.float64, order="F", copy=True)

    q, _, info = orgqr(q_input, tau, overwrite_a=1)

    if info < 0:
        raise ValueError(f"LAPACK ORGQR rejected argument {-info}.")
    if info > 0:
        raise np.linalg.LinAlgError(
            f"LAPACK ORGQR failed with INFO={info}."
        )

    if not np.isfinite(q).all() or not np.isfinite(r).all():
        raise FloatingPointError(
            "QR factorization produced non-finite values."
        )

    return np.asarray(q, dtype=np.float64), np.asarray(r, dtype=np.float64)
