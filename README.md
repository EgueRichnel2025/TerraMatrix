# TerraMatrix

**High-Performance Earth Observation Analytics**

TerraMatrix is a scientific platform for analyzing satellite data through
high-performance numerical linear algebra.

The project investigates how BLAS, LAPACK, OpenBLAS, and Intel oneMKL
can accelerate matrix-based Earth observation workloads while preserving
numerical accuracy and scientific reproducibility.

## Scientific objectives

- Process real satellite observations.
- Build matrices from multispectral raster data.
- Implement and evaluate BLAS/LAPACK operations.
- Study PCA, SVD, QR factorization, and numerical conditioning.
- Compare OpenBLAS and Intel oneMKL under controlled conditions.
- Measure execution time, parallel speedup, and numerical accuracy.
- Produce interpretable geospatial analyses.

## Planned applications

- Vegetation monitoring and spectral anomaly analysis.
- Before-and-after comparisons following disasters.
- Land-cover change and urbanization monitoring.

Detected anomalies and changes are indicators for further investigation.
They are not, by themselves, diagnoses of agricultural conditions or proof
of a particular cause.

## Technology

- Python 3.12
- NumPy and SciPy
- BLAS and LAPACK
- OpenBLAS and Intel oneMKL
- pytest and Ruff
- Streamlit (planned interface)

## Scientific integrity

All reported benchmark results must come from actual measurements.
Experiments must record their computational environment, input dimensions,
backend, thread count, numerical validation, and relevant data metadata.

## Development status

**Phase 0 — Foundations**

The project is being developed incrementally. Satellite-data processing,
the numerical engine, scientific analyses, benchmarks, and interface
will be implemented and tested in subsequent phases.

## Development setup

Create and activate a virtual environment, then install the project:

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest -q
```

Check code quality:

```bash
ruff check .
```

## Project structure

```text
TerraMatrix/
├── docs/
├── src/
│   └── terramatrix/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

The architecture and directory structure will evolve as functional
components are implemented.
