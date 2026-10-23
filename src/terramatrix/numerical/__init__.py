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
from terramatrix.numerical.least_squares import (
    solve_normal_equations,
    solve_pivoted_qr,
    solve_qr,
    solve_svd,
)
from terramatrix.numerical.pca import PCAResult, principal_component_analysis

__all__ = [
    "MatrixDiagnostics",
    "NumericalLibrary",
    "PCAResult",
    "classify_backend",
    "dgemm",
    "dgemv",
    "diagnose_matrix",
    "dsyrk",
    "get_numerical_runtime_info",
    "principal_component_analysis",
    "qr_factorization",
    "solve_normal_equations",
    "solve_pivoted_qr",
    "solve_qr",
    "solve_svd",
    "svd_factorization",
    "symmetric_eigendecomposition",
]
