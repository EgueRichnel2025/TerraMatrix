"""Diagnostics for numerical libraries loaded by the TerraMatrix process."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy
import scipy
from threadpoolctl import threadpool_info


@dataclass(frozen=True, slots=True)
class NumericalLibrary:
    """Information reported for a loaded native numerical library."""

    name: str
    user_api: str | None
    internal_api: str | None
    version: str | None
    num_threads: int | None
    threading_layer: str | None
    architecture: str | None
    prefix: str | None
    filepath: str | None


def classify_backend(info: Mapping[str, Any]) -> str:
    """Classify a native library using its reported identification fields."""
    internal_api = str(info.get("internal_api") or "").lower()
    user_api = str(info.get("user_api") or "").lower()

    evidence = " ".join(
        str(info.get(field) or "").lower()
        for field in ("internal_api", "prefix", "filepath")
    )

    if "openblas" in evidence:
        return "OpenBLAS"

    if "mkl" in evidence:
        return "Intel MKL / oneMKL"

    if "flexiblas" in evidence:
        return "FlexiBLAS"

    if "blis" in evidence:
        return "BLIS"

    if "accelerate" in evidence or "veclib" in evidence:
        return "Apple Accelerate"

    if "atlas" in evidence:
        return "ATLAS"

    if user_api == "blas":
        return "Unknown BLAS backend"

    if user_api == "openmp" or internal_api == "openmp":
        return "OpenMP runtime"

    if internal_api:
        return internal_api

    return "Unknown native library"


def _optional_string(value: Any) -> str | None:
    """Convert an available value to a string without inventing missing data."""
    if value is None:
        return None

    return str(value)


def _optional_int(value: Any) -> int | None:
    """Return an integer value when the library reports one."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value

    return None


def _build_library(info: Mapping[str, Any]) -> NumericalLibrary:
    """Convert a threadpoolctl record into a validated data structure."""
    return NumericalLibrary(
        name=classify_backend(info),
        user_api=_optional_string(info.get("user_api")),
        internal_api=_optional_string(info.get("internal_api")),
        version=_optional_string(info.get("version")),
        num_threads=_optional_int(info.get("num_threads")),
        threading_layer=_optional_string(info.get("threading_layer")),
        architecture=_optional_string(info.get("architecture")),
        prefix=_optional_string(info.get("prefix")),
        filepath=_optional_string(info.get("filepath")),
    )


def get_numerical_runtime_info() -> dict[str, object]:
    """Report environment details and numerical libraries currently detected.

    The result describes the current Python process. It does not guarantee
    that a particular library performs every numerical operation when
    several compatible libraries are loaded simultaneously.
    """
    libraries = [_build_library(info) for info in threadpool_info()]

    blas_libraries = [
        library
        for library in libraries
        if library.user_api is not None and library.user_api.lower() == "blas"
    ]

    blas_backends = sorted({library.name for library in blas_libraries})

    if not blas_libraries:
        backend_status = "no_blas_library_detected"
    elif len(blas_libraries) > 1 and len(blas_backends) > 1:
        backend_status = "multiple_blas_backends_detected"
    elif len(blas_libraries) > 1:
        backend_status = "multiple_blas_libraries_detected"
    else:
        backend_status = "single_blas_library_detected"

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "blas_backend_status": backend_status,
        "blas_backends": blas_backends,
        "blas_library_count": len(blas_libraries),
        "libraries": [asdict(library) for library in libraries],
    }
