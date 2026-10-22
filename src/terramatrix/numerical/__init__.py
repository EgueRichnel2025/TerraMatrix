"""Numerical operations and runtime diagnostics for TerraMatrix."""

from terramatrix.numerical.backend import (
    NumericalLibrary,
    classify_backend,
    get_numerical_runtime_info,
)
from terramatrix.numerical.blas import dgemm

__all__ = [
    "NumericalLibrary",
    "classify_backend",
    "dgemm",
    "get_numerical_runtime_info",
]
