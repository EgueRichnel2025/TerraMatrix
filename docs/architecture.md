# TerraMatrix — Architecture

## 1. Purpose

TerraMatrix is a scientific platform for Earth observation analytics.
Its central objective is to demonstrate how high-performance numerical
linear algebra contributes to the processing and analysis of satellite data.

The application must connect real scientific workloads to identifiable
BLAS/LAPACK operations and reproducible performance measurements.

## 2. Target architecture

```text
Satellite Data
      |
      v
Data Pipeline
      |
      v
Application Layer <------> Streamlit Interface
      |
      v
Analysis Engine
      |
      v
Numerical Engine
      |
      v
BLAS / LAPACK
      |
      +------------------+
      |                  |
      v                  v
  OpenBLAS           Intel oneMKL
```

## 3. Data pipeline

The data pipeline is responsible for:

- Discovering and selecting satellite scenes.
- Recording acquisition metadata and spatial information.
- Reading the required raster bands.
- Handling invalid pixels and missing values.
- Checking spatial alignment and resolution.
- Preparing numerical arrays and spectral matrices.
- Preserving the geospatial information required to interpret results.

For a multispectral scene, the analysis matrix is represented as
X in R^(N x B), where N is the number of selected pixels and B
is the number of spectral bands.

The matrix construction process must document the pixel selection,
band order, data type, preprocessing, and handling of invalid values.

## 4. Application layer

The application layer coordinates the execution of user workflows.

Its responsibilities include:

- Validating user inputs.
- Loading configuration.
- Calling the appropriate analysis service.
- Coordinating data preparation and numerical computation.
- Returning results and execution metadata.

Numerical algorithms must not be implemented directly in the
interface layer.

## 5. Analysis engine

The analysis engine implements scientific workflows by combining
reusable numerical operations.

Planned capabilities include:

- Principal component analysis (PCA).
- Spectral anomaly analysis.
- Statistical summaries.
- Before-and-after change detection.
- Temporal and land-cover change analysis.

Analysis methods must state their assumptions and limitations.
Detected anomalies or changes are indicators for further investigation,
not automatic diagnoses of agricultural conditions or proof of causation.

## 6. Numerical engine

The numerical engine exposes identifiable, testable numerical operations.

### BLAS

- DGEMM: general matrix-matrix multiplication.
- DSYRK: symmetric rank-k update, including covariance construction.
- DGEMV: general matrix-vector multiplication.

### LAPACK

- QR factorization.
- Singular value decomposition (SVD).
- Symmetric eigendecomposition.
- Numerical rank and conditioning diagnostics.

The implementation must make the relationship between scientific
algorithms and their underlying numerical operations understandable.

High-level NumPy or SciPy routines may be used when appropriate,
provided that their numerical behavior and actual backend are verified
when those details matter to an experiment.

## 7. PCA strategy

For a centered data matrix X_c, PCA can be performed using either
a covariance-based approach or a direct SVD.

### Covariance-based approach

Construct the covariance matrix from X_c^T X_c and compute its
symmetric eigendecomposition.

This approach can be attractive when the number of spectral bands
is small relative to the number of observations.

However, forming the Gram matrix squares the condition number of
the centered data matrix when the relevant condition numbers are defined.
Consequently, numerical accuracy must be evaluated.

### Direct SVD

Compute the factorization:

X_c = U Sigma V^T

The singular values and right singular vectors can be used to obtain
principal components and explained variance.

The two approaches must be validated and compared on suitable inputs.
The choice must depend on the dimensions, numerical requirements,
and measured behavior of the problem.

## 8. Backend isolation

OpenBLAS and Intel oneMKL must be evaluated in controlled environments.

For every experiment, TerraMatrix must identify the numerical backend
actually loaded. A manually selected backend label is not sufficient
evidence of which library performed the computation.

Experiments must record, where available:

- Backend name and version.
- Python, NumPy, and SciPy versions.
- Operating system and processor information.
- Thread count and threading layer.
- Matrix dimensions, data type, and algorithm.
- Execution time and repetition count.
- Numerical validation results.

Configurations must not be silently mixed within a comparison.

## 9. Benchmark methodology

Benchmarks must use documented input sizes and controlled parameters.

Where appropriate, measurements should include:

- Warm-up runs.
- Multiple timed repetitions.
- Median and variability of execution time.
- Speedup relative to a measured baseline.
- Parallel efficiency.
- Memory usage, when reliably measurable.
- Accuracy relative to an appropriate reference.

Speedup and efficiency must be calculated from actual measurements.
No performance result may be invented or presented as universal.

## 10. Scientific validation

Tests must cover:

- Valid matrices and expected dimensions.
- Incorrect dimensions and invalid inputs.
- Missing or non-finite values.
- Singular and rank-deficient matrices.
- Ill-conditioned numerical problems.
- Numerical agreement with trusted reference results.
- Reproducibility under documented conditions.

Geospatial results must preserve enough metadata to connect numerical
outputs to the pixels, bands, dates, and region being analyzed.

## 11. Interface

Streamlit is the planned interface for selecting data, configuring
analyses, displaying maps, and presenting results.

The interface should make scientific workflows accessible without
requiring the user to understand BLAS, LAPACK, or thread configuration.

Advanced performance information will be presented separately from
the essential analytical results.

## 12. Implementation roadmap

Development will proceed incrementally:

1. Foundations and environment.
2. Satellite data discovery and preprocessing.
3. Numerical engine and backend identification.
4. PCA and numerical validation.
5. Agriculture analysis.
6. Disaster-related change detection.
7. Land-cover and temporal analysis.
8. Controlled performance benchmarks.
9. Interface integration and reproducibility.
10. Packaging, deployment, and documentation.

New modules will be added when useful functionality is ready.
The architecture must evolve based on tested implementation decisions,
not simply to populate directories.
