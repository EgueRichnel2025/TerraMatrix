import terramatrix.numerical.backend as backend_module
from terramatrix.numerical import (
    classify_backend,
    get_numerical_runtime_info,
)


def test_classify_openblas() -> None:
    info = {
        "user_api": "blas",
        "internal_api": "openblas",
        "version": "0.3.34",
    }

    assert classify_backend(info) == "OpenBLAS"


def test_classify_intel_mkl() -> None:
    info = {
        "user_api": "blas",
        "internal_api": "mkl",
        "version": "2025.0",
    }

    assert classify_backend(info) == "Intel MKL / oneMKL"


def test_classify_unknown_blas() -> None:
    info = {
        "user_api": "blas",
        "internal_api": "unknown_backend",
    }

    assert classify_backend(info) == "Unknown BLAS backend"


def test_runtime_info_reports_detected_blas(monkeypatch) -> None:
    fake_libraries = [
        {
            "user_api": "blas",
            "internal_api": "openblas",
            "version": "0.3.test",
            "num_threads": 4,
            "threading_layer": "pthreads",
            "architecture": "Haswell",
            "prefix": "libopenblas",
            "filepath": "openblas-test.dll",
        }
    ]

    monkeypatch.setattr(
        backend_module,
        "threadpool_info",
        lambda: fake_libraries,
    )

    result = get_numerical_runtime_info()

    assert result["blas_backend_status"] == "single_blas_library_detected"
    assert result["blas_backends"] == ["OpenBLAS"]
    assert result["blas_library_count"] == 1
    assert result["python_version"]
    assert result["numpy_version"]
    assert result["scipy_version"]

    libraries = result["libraries"]
    assert len(libraries) == 1
    assert libraries[0]["num_threads"] == 4
    assert libraries[0]["version"] == "0.3.test"


def test_runtime_info_detects_multiple_blas_backends(monkeypatch) -> None:
    fake_libraries = [
        {
            "user_api": "blas",
            "internal_api": "openblas",
            "num_threads": 4,
        },
        {
            "user_api": "blas",
            "internal_api": "mkl",
            "num_threads": 4,
        },
    ]

    monkeypatch.setattr(
        backend_module,
        "threadpool_info",
        lambda: fake_libraries,
    )

    result = get_numerical_runtime_info()

    assert result["blas_backend_status"] == "multiple_blas_backends_detected"
    assert set(result["blas_backends"]) == {
        "OpenBLAS",
        "Intel MKL / oneMKL",
    }
    assert result["blas_library_count"] == 2


def test_runtime_info_handles_missing_blas(monkeypatch) -> None:
    fake_libraries = [
        {
            "user_api": "openmp",
            "internal_api": "openmp",
            "num_threads": 4,
        }
    ]

    monkeypatch.setattr(
        backend_module,
        "threadpool_info",
        lambda: fake_libraries,
    )

    result = get_numerical_runtime_info()

    assert result["blas_backend_status"] == "no_blas_library_detected"
    assert result["blas_backends"] == []
    assert result["blas_library_count"] == 0
