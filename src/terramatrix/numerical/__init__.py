"""Numerical operations and runtime diagnostics for TerraMatrix."""

from terramatrix.numerical.backend import (
    NumericalLibrary,
    classify_backend,
    get_numerical_runtime_info,
)
from terramatrix.numerical.blas import dgemm, dgemv, dsyrk
from terramatrix.numerical.lapack import qr_factorization, svd_factorization

__all__ = [
    "NumericalLibrary",
    "classify_backend",
    "dgemm",
    "dgemv",
    "dsyrk",
    "get_numerical_runtime_info",
    "qr_factorization",
    "svd_factorization",
]
