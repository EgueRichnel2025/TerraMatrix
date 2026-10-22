"""Numerical operations and runtime diagnostics for TerraMatrix."""

from terramatrix.numerical.backend import (
    NumericalLibrary,
    classify_backend,
    get_numerical_runtime_info,
)

__all__ = [
    "NumericalLibrary",
    "classify_backend",
    "get_numerical_runtime_info",
]
