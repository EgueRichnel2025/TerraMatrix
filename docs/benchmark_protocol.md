# TerraMatrix — Benchmark Protocol

## 1. Purpose

This document defines the protocol for evaluating the performance,
numerical accuracy, and scalability of TerraMatrix numerical workloads.

The objective is to measure real execution behavior under documented
conditions. Benchmark results must be reproducible to the extent allowed
by the hardware, operating system, numerical libraries, and input data.

This protocol specifies how experiments must be conducted. It does not
contain measured performance claims.

## 2. Experimental principles

Every benchmark must follow these principles:

- Use identical inputs when comparing numerical backends.
- Keep the algorithm, data type, dimensions, and memory layout consistent.
- Identify the numerical libraries actually loaded.
- Record thread configuration and relevant hardware information.
- Validate numerical outputs independently of timing measurements.
- Measure multiple repetitions and report variability.
- Preserve raw measurements and experiment metadata.
- Never invent timings, speedups, or memory measurements.
- Document unavailable measurements and experimental limitations.

Results from different machines or configurations must not be combined
as though they came from a controlled comparison.

## 3. Benchmark categories

### 3.1. BLAS operations

The initial operation-level benchmarks are:

- DGEMM: general matrix-matrix multiplication.
- DSYRK: symmetric rank-k update.
- DGEMV: general matrix-vector multiplication.

For every operation, record the dimensions, data type, memory layout,
transpose settings, input initialization procedure, and output validation.

Dense square-matrix workloads and satellite-shaped workloads must be
reported separately.

### 3.2. Numerical methods

The method-level benchmarks cover:

- QR factorization.
- Singular value decomposition (SVD).
- Symmetric eigendecomposition (EIGH).
- Principal component analysis (PCA).

The benchmark must include the cost of the selected method and identify
whether preprocessing or matrix construction is included in the timing.

### 3.3. End-to-end scientific workloads

Pipeline benchmarks evaluate complete analytical tasks, not just
isolated numerical operations.

A pipeline measurement must state whether the timing includes:

- Input loading.
- Raster preprocessing.
- Masking and spatial alignment.
- Matrix construction.
- Numerical computation.
- Result generation.

A separate end-to-end measurement may include all these stages.
The scope of each measurement must be explicit.

## 4. Input workloads

### 4.1. Synthetic workloads

Synthetic matrices provide controlled inputs for testing dimensions,
conditioning, numerical rank, and computational cost.

Each generated input must record its dimensions, data type, generation
method, random seed, and relevant numerical characteristics.

The initial dense-matrix sizes must be selected according to the actual
memory capacity and performance of the test machine. Larger sizes must
not be attempted blindly.

For a dense matrix with dimensions m x n stored in float64, the array
requires approximately 8mn bytes, excluding temporary arrays and
additional program overhead.

A benchmark involving multiple matrices must estimate total memory
requirements before allocating its inputs.

### 4.2. Satellite-shaped workloads

Let X have dimensions N x B, where N is the number of selected pixels
and B is the number of spectral bands.

For satellite workloads, vary N and, when scientifically justified, B.
Record the selection criteria for pixels and bands.

The dimensions of X must not be confused with the dimensions of the
covariance matrix, which is B x B.

Synthetic dense-matrix benchmarks and satellite-shaped workloads answer
different questions and must have separate results.

### 4.3. Real satellite datasets

When real data becomes available, record at least:

- Dataset identifier and source.
- Acquisition date and geographic region.
- Selected spectral bands.
- Raster dimensions and spatial resolution.
- No-data and invalid-pixel handling.
- Preprocessing and alignment procedures.
- Data type and resulting matrix dimensions.

Before-and-after datasets must use compatible spatial grids, band
definitions, and preprocessing procedures.

## 5. Backend isolation

OpenBLAS and Intel oneMKL must be evaluated in controlled environments.

For each run, record the libraries detected in the executing Python
process, including their reported names, versions, paths, and thread
configuration when available.

A configuration preference is not proof that a particular backend
was used.

If multiple BLAS implementations are loaded, report the situation rather
than silently attributing an operation to one library.

Prefer separate, documented environments for OpenBLAS and oneMKL.
Do not assume that selecting a thread count also selects the backend.

If oneMKL is not available or cannot be verified, mark that experiment
as unavailable. Never substitute another backend's measurements under
the oneMKL label.

## 6. Threading experiments

Threading experiments must record the requested and reported thread
counts, along with the backend configuration.

Use controlled thread-count experiments where the environment supports
them. The tested counts must be adapted to the actual machine.

Avoid concurrent benchmark runs that compete for the same CPU resources.

For a given workload and backend, define the measured baseline T(1) as
the execution time using one thread under the same experimental protocol.

For p threads, calculate:

Speedup(p) = T(1) / T(p)

Parallel efficiency is:

Efficiency(p) = Speedup(p) / p

These quantities must use measured execution times. They must not be
derived from assumed performance or results from another configuration.

Speedup and efficiency are workload-specific. Increasing the number of
threads does not guarantee faster execution.

## 7. Timing protocol

For each benchmark configuration:

1. Prepare and validate the input.
2. Initialize required arrays outside the timed region unless allocation
   is explicitly part of the workload.
3. Perform the documented warm-up runs.
4. Execute the timed repetitions using a monotonic high-resolution timer.
5. Store the raw duration of each repetition.
6. Validate numerical outputs separately.
7. Calculate and report summary statistics.

The initial configuration uses five timed repetitions and one warm-up
run. Increase the repetition count when measurements are noisy or
the workload is too short for reliable timing.

Report at least the median execution time. Report variability, such as
the interquartile range, whenever enough measurements are available.

Record whether data loading, memory allocation, preprocessing, and
output writing are included in each timed region.

Avoid concurrent workloads and document relevant power-management,
system-load, or thermal conditions when they may affect the results.

## 8. Numerical validation

Performance is meaningful only when the numerical result is acceptable.

Validation must be adapted to the operation and data type.

Examples include:

- DGEMM: compare the computed product with a reference result on
  appropriate test inputs.
- DSYRK: validate symmetry and agreement with the reference product.
- DGEMV: compare the computed vector with a reference result.
- QR: assess reconstruction residuals and factor orthogonality.
- SVD: assess reconstruction residuals and singular-value agreement.
- EIGH: assess eigenpair residuals and orthogonality.
- PCA: assess explained variance, reconstruction error, and consistency
  of the resulting components.

For a nonzero reference matrix A, a relative Frobenius reconstruction
error may be reported as:

||A - A_approx||_F / ||A||_F

Validation tolerances must reflect floating-point precision, dimensions,
conditioning, and the numerical method.

Numerical comparisons must account for legitimate differences due to
rounding, particularly when singular values or eigenvalues are clustered.

A failed numerical validation must be reported even when execution time
is favorable.

## 9. Memory measurements

Memory usage should be measured only when the selected method provides
reliable measurements.

Record the measurement method, platform, and whether the result refers
to process memory, peak memory, or an estimate of array storage.

Do not confuse the theoretical storage of input matrices with total
runtime memory consumption.

If a trustworthy measurement is unavailable, report it as unavailable
rather than inventing a value.

## 10. Comparison rules

A direct backend comparison must use:

- The same machine and operating environment where feasible.
- The same validated input data.
- The same algorithm and data type.
- The same matrix dimensions and memory layout.
- The same measurement protocol.
- Recorded thread configurations.
- Equivalent numerical validation requirements.

Each backend must be evaluated using its own identified runtime
configuration.

Cross-machine comparisons must be labeled separately and must include
the relevant hardware and software differences.

Do not compare isolated-operation timings with end-to-end pipeline
timings as if they measured the same workload.

## 11. Metrics and reporting

Each experiment should report:

- Experiment identifier.
- Date and time in UTC.
- Dataset or synthetic-input identifier.
- Matrix dimensions and data type.
- Numerical operation or algorithm.
- Backend identity and library version.
- Requested and reported thread counts.
- Number of warm-up runs and timed repetitions.
- Raw execution times.
- Median execution time and variability.
- Numerical validation results.
- Memory measurements, when available.
- Hardware and software environment.
- Warnings, failures, and limitations.

When a baseline exists, report the measured speedup and, where
appropriate, parallel efficiency.

Plots and summary tables must be generated from stored measurements.

## 12. Result storage and reproducibility

Raw measurements must be retained separately from derived summaries.

Each stored experiment must link to its configuration and relevant
environment metadata.

Record the source-control commit where available. A working-tree
modification must be disclosed when the experiment is run against
uncommitted source code.

A result must not be described as fully reproducible if required inputs,
configuration, environment information, or raw measurements are missing.

The same experiment identifier must not silently overwrite results
from a different configuration.

## 13. Initial experiment sequence

The initial experimental sequence is:

1. Verify the numerical backend and environment metadata.
2. Validate correctness on small deterministic inputs.
3. Benchmark DGEMM, DSYRK, and DGEMV.
4. Benchmark QR, SVD, and symmetric eigendecomposition.
5. Evaluate thread scaling for selected workloads.
6. Evaluate PCA with numerical validation.
7. Evaluate complete analysis pipelines.
8. Repeat relevant workloads with real satellite data when available.
9. Compare backends only after both configurations are independently
   verified.
10. Publish summaries together with the experiment metadata and
    methodological limitations.

All benchmark results must originate from actual executions.
