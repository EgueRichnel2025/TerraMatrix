"""Numerical operations and runtime diagnostics for TerraMatrix."""

from terramatrix.numerical.backend import (
    NumericalLibrary,
    classify_backend,
    get_numerical_runtime_info,
)
from terramatrix.numerical.blas import dgemm, dgemv, dsyrk
from terramatrix.numerical.diagnostics import MatrixDiagnostics, diagnose_matrix
from terramatrix.numerical.lapack import (
    qr_factorization,
    svd_factorization,
    symmetric_eigendecomposition,
)

__all__ = [
    "MatrixDiagnostics",
    "NumericalLibrary",
    "classify_backend",
    "dgemm",
    "dgemv",
    "diagnose_matrix",
    "dsyrk",
    "get_numerical_runtime_info",
    "qr_factorization",
    "svd_factorization",
    "symmetric_eigendecomposition",
]
