# TerraMatrix — Experimental Catalogue

## 1. Purpose

This catalogue defines the initial numerical experiments for TerraMatrix.

It complements the benchmark protocol by specifying the scientific
questions, input characteristics, parameters, methods, and measurements
required for each experiment.

This document defines planned experiments. It does not contain fabricated
execution times, measured errors, or benchmark conclusions.

Every experiment must record its actual environment and configuration.

## 2. General experimental conventions

Unless an experiment specifies otherwise, use the following conventions.

| Parameter | Initial value or rule |
|---|---|
| Floating-point type | float64 |
| Random seed | 20261022 |
| Number of observations, n | 1000 |
| Number of features, p | 20 |
| Response noise standard deviation | 1e-3 |
| Number of timed repetitions | 5 |
| Warm-up runs | 1 |
| Baseline backend | Actually detected and recorded |
| Reference results | Independently computed and validated |

These values define initial experimental conditions. They are not measured
results and may be adjusted when memory, runtime, or numerical requirements
justify a change.

The random generator, its seed, the matrix-generation algorithm, and the
order of operations must be recorded.

When comparing methods, use the same input matrices and response vectors.

## 3. Experiment TM-LS-01 — Well-conditioned, full-rank matrix

### Scientific question

How do normal equations, QR factorization, and SVD compare on a
full-rank least-squares problem with a well-conditioned design matrix?

### Input construction

Construct a real matrix X with dimensions n x p, where n >= p.

The intended matrix must have full column rank and a moderate 2-norm
condition number.

For controlled experiments, an orthogonal-factor construction may be used:

X = Q Sigma V^T

where Q has orthonormal columns, V is orthogonal, and Sigma contains
the prescribed positive singular values.

The generated matrix must be checked numerically. Its measured
condition number and numerical rank must be recorded.

Generate a known coefficient vector beta_true using the documented
random generator. Construct the response vector:

b = X beta_true + epsilon

where epsilon is the documented noise vector.

### Initial parameters

| Parameter | Value |
|---|---|
| Experiment ID | TM-LS-01 |
| n | 1000 |
| p | 20 |
| Data type | float64 |
| Random seed | 20261022 |
| Noise standard deviation | 1e-3 |
| Collinearity epsilon | Not applicable |
| Rank | Expected to be p |
| Conditioning | Full rank with moderate condition number |

The measured condition number must confirm whether the matrix meets the
intended conditioning category.

### Methods

- Normal equations: solve X^T X beta = X^T b.
- QR-based least-squares solution.
- SVD-based least-squares solution.

The normal-equations method is included as a comparison baseline, not
as a universally recommended solution.

### Expected properties

- The system has a unique least-squares solution when X has full column
  rank.
- All three methods should produce acceptable residuals when implemented
  and configured correctly.
- Numerical differences may occur because the methods have different
  floating-point behavior.
- Execution time and numerical error must be measured, not assumed.

## 4. Experiment TM-LS-02 — Increasing ill-conditioning

### Scientific question

How does increasing near-collinearity affect numerical rank diagnostics,
least-squares residuals, coefficient recovery, and the behavior of
normal equations, QR, and SVD?

### Input construction

Generate a reproducible full-rank design matrix whose singular values
span a controlled range.

One suitable construction is:

X = Q Sigma V^T

where Q has orthonormal columns, V is orthogonal, and Sigma contains
positive singular values with a prescribed range.

Alternatively, a column-perturbation construction may be used to study
near-collinearity directly. The selected construction must be documented.

For the controlled singular-spectrum construction, define:

kappa_2(X) = sigma_max(X) / sigma_min(X)

and use prescribed target condition numbers to create increasingly
ill-conditioned cases.

For a column-perturbation construction, record collinearity_epsilon
and measure the resulting condition number rather than assuming that
epsilon alone determines it exactly.

### Initial parameters

| Parameter | Values |
|---|---|
| Experiment ID | TM-LS-02 |
| n | 1000 |
| p | 20 |
| Data type | float64 |
| Random seed | 20261022 |
| Noise standard deviation | 1e-3 |
| Target condition numbers, controlled construction | 1e2, 1e4, 1e6 |
| Collinearity epsilon, perturbation construction | 1e-2, 1e-4, 1e-6 |
| Rank | Intended to be p for nonzero prescribed singular values |

The singular-spectrum and column-perturbation constructions are alternative
experimental families. They must not be silently mixed in the same series.

Record the actual condition number and numerical rank of every matrix.

### Methods

- Normal equations.
- QR-based least-squares solution.
- SVD-based least-squares solution.

Use identical input matrices and response vectors for all methods.

### Expected properties

- Increasing the condition number makes the problem more sensitive to
  perturbations.
- Forming X^T X squares the 2-norm condition number for full-column-rank X.
- Normal equations can consequently lose accuracy more rapidly on
  ill-conditioned problems.
- QR and SVD have different numerical properties and must be evaluated
  empirically.
- The observed numerical rank may depend on the method and its tolerance.

Do not claim that a method has failed merely because its coefficients
differ slightly from beta_true. Interpret coefficient error together
with conditioning, noise, residuals, and the numerical method.

## 5. Experiment TM-LS-03 — Exactly rank-deficient matrix

### Scientific question

How do numerical methods behave when the design matrix is exactly
rank-deficient and the least-squares coefficients are not uniquely
determined?

### Input construction

Generate a reproducible matrix with n rows and p columns, then construct
an exact duplicate column, for example by assigning the final column
to be identical to the first.

The duplicate must be created by an exact copy operation rather than
by adding a small perturbation.

The constructed matrix is mathematically rank-deficient. Numerical rank
diagnostics must still record the tolerance and computed rank.

Generate beta_true and the response vector using the documented random
generator and noise procedure.

### Initial parameters

| Parameter | Value |
|---|---|
| Experiment ID | TM-LS-03 |
| n | 1000 |
| p | 20 |
| Data type | float64 |
| Random seed | 20261022 |
| Noise standard deviation | 1e-3 |
| Collinearity epsilon | 0 for the exact duplicate-column construction |
| Maximum expected algebraic rank | p - 1 |

An exactly duplicated column makes the mathematical matrix rank-deficient.
The reported numerical rank must also include the method and tolerance
used to determine it.

### Methods

- Normal equations, with solver status and warnings recorded.
- QR factorization, distinguishing ordinary QR from rank-revealing QR.
- SVD-based least-squares solution, including its rank tolerance.

The experiment must not assume that every method returns a unique
coefficient vector.

### Expected properties

- The least-squares coefficients are not uniquely determined when the
  design matrix is rank-deficient.
- The matrix X^T X is singular in exact arithmetic.
- A solver may reject the normal-equations system, return a result with
  a warning, or behave according to its implementation and tolerance.
- Rank-revealing QR and SVD can identify rank deficiency when configured
  with suitable diagnostics.
- An SVD-based pseudoinverse solution can provide a minimum-norm
  least-squares solution when its cutoff is appropriately defined.

For this case, do not evaluate coefficient recovery alone. Record the
rank, solver status, residual norm, and whether the reported solution
satisfies the least-squares conditions.

## 6. Measurements required for every experiment

Each execution must record:

- Experiment ID and timestamp in UTC.
- Matrix dimensions and floating-point type.
- Matrix-generation method and random seed.
- Noise parameters.
- Collinearity parameter or prescribed singular spectrum.
- Measured numerical rank and rank tolerance.
- Measured condition number when available.
- Backend identity and version.
- Python, NumPy, and SciPy versions.
- Thread count and threading configuration.
- Method and solver settings.
- Warm-up count and timed repetition count.
- Raw execution times and median execution time.
- Residual norm and relative residual.
- Coefficient error when the experiment makes it meaningful.
- Solver warnings, failures, and status.
- Relevant environment metadata and source-control commit.

The recorded parameters must describe the actual execution, not merely
the requested configuration.

## 7. Numerical metrics

### Absolute residual norm

For the computed coefficient vector beta_hat, record:

||X beta_hat - b||_2

### Relative residual

When b is nonzero, record:

||X beta_hat - b||_2 / ||b||_2

### Relative coefficient error

When beta_true is known and the problem makes coefficient recovery
meaningful, record:

||beta_hat - beta_true||_2 / ||beta_true||_2

This metric must be interpreted with conditioning and response noise.
For a rank-deficient matrix, coefficients may be non-unique, so
coefficient error alone is not a sufficient correctness measure.

### Condition number and rank

Record the condition number, numerical rank, and rank threshold used.
For rank-deficient or ill-conditioned problems, document the limitations
of these diagnostics.

### Timing statistics

Retain all measured durations. Report the median and an appropriate
measure of variability.

Calculate speedup only from valid measurements of the same workload
under controlled configurations.

## 8. Correctness checks

Before accepting benchmark results:

1. Verify all input dimensions.
2. Check that generated data are finite.
3. Confirm the expected structural properties of the matrix.
4. Record the actual numerical rank and condition number.
5. Validate the solver output with residual and method-specific checks.
6. Record warnings or failures rather than silently discarding them.
7. Ensure all compared methods used the same input data.
8. Confirm the backend and thread configuration for the execution.

Reference results and tolerances must be documented. No result may be
marked successful solely because the solver returned without raising
an exception.

## 9. Result interpretation

Interpretation must distinguish the following:

- A difficult mathematical problem from an implementation defect.
- Small floating-point differences from unacceptable numerical error.
- A small residual from accurate coefficient recovery.
- A measured runtime difference from a general performance guarantee.
- Mathematical rank from numerically estimated rank.

The experiments must report both favorable and unfavorable results.

## 10. Experimental status

The catalogue defines planned experiments and expected mathematical
properties. No execution times, measured errors, or performance rankings
are claimed in this document.

After implementation, each experiment must produce machine-readable
raw measurements and metadata before conclusions are written.

Actual observations must be reported separately from theoretical
expectations. Any deviation from these specifications must be recorded
with its justification.
