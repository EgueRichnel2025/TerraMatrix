# TerraMatrix — PCA Experimental Results

## 1. Scope

This document records the observed numerical results from the initial
synthetic PCA comparison. It complements `experiments.md`, which defines
the planned experimental protocol.

These measurements were produced by the local TerraMatrix execution.
They describe three synthetic matrices and must not be interpreted as
universal guarantees about PCA algorithms.

Raw reports are preserved separately under `results/experiments/`.

## 2. Execution configuration

| Parameter | Observed configuration |
|---|---|
| Experiment suite | TM-PCA-COMPARISON-01 |
| Data type | float64 |
| Number of observations | 1000 |
| Number of features | 20 |
| Base seed | 20261022 |
| Case seeds | 20261022, 20261023, 20261024 |
| Python | 3.12.8 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| Operating system | Windows 11 |
| Detected numerical backend | OpenBLAS |
| Loaded BLAS library count | 2 |
| Reported thread count per detected library | 12 |
| Timings collected | No |

The loaded libraries were reported as OpenBLAS `0.3.34.106.0`
and OpenBLAS `0.3.31.dev`. The environment detector reported
`multiple_blas_libraries_detected`. The observations do not establish
that both libraries executed every operation.

## 3. Results summary

| Experiment | Measured condition number | Numerical rank | Maximum relative singular-value error on significant components | Relative global explained-variance difference |
|---|---:|---:|---:|---:|
| TM-PCA-01 — Well-conditioned | 2.0 | 20/20 | 1.289092e-15 | 1.411691e-15 |
| TM-PCA-02 — Ill-conditioned | 1.0e6 | 20/20 | 3.685965e-06 | 5.369568e-16 |
| TM-PCA-03 — Rank-deficient | Infinite under the numerical-rank diagnostic | 19/20 | 1.153019e-15 | 1.072095e-15 |

The singular-value reference was the direct-SVD result. Components were
treated as significant when their reference singular value exceeded
`max(m, n) * eps(float64) * sigma_max`.

Relative errors were not calculated for reference components below this
threshold.

## 4. Reconstruction and orthogonality

Observed relative Frobenius reconstruction errors:

| Experiment | Covariance PCA | Direct-SVD PCA |
|---|---:|---:|
| TM-PCA-01 | 1.450818e-15 | 1.567130e-15 |
| TM-PCA-02 | 1.660761e-15 | 5.250295e-16 |
| TM-PCA-03 | 1.013054e-15 | 1.314212e-15 |

The observed component-orthogonality errors were also on the order of
1e-15 in these runs.

These results indicate close reconstruction accuracy for the three
tested matrices. They do not establish equal numerical accuracy for
all matrices or condition numbers.

## 5. Small singular values

### TM-PCA-01 — Well-conditioned matrix

The measured condition number was approximately 2. Both methods
returned a smallest singular value of 0.5.

The maximum relative difference between significant singular values
was approximately 1.29e-15.

### TM-PCA-02 — Ill-conditioned matrix

The measured condition number was approximately 1e6.

The smallest singular value was approximately:

- Covariance PCA: 1.000004e-6.
- Direct SVD: 1.000000e-6.

The maximum relative singular-value difference among significant
components was approximately 3.69e-6, even though the relative global
explained-variance difference was approximately 5.37e-16.

This run demonstrates that a global variance metric can conceal
larger relative differences in small singular values.

### TM-PCA-03 — Exactly rank-deficient matrix

The final column was an exact copy of the first column. Both methods
reported numerical rank 19 out of 20 under the SVD-based reference
rank tolerance.

The covariance method used an absolute covariance-eigenvalue tolerance
of approximately 9.572659e-15. Its smallest returned singular value
was zero after clipping the corresponding eigenvalue.

Direct SVD returned a smallest singular value of approximately
4.825461e-15, consistent with a numerically zero singular value at
the selected reference tolerance.

The covariance threshold is an eigenvalue threshold; it must not be
confused with the singular-value rank tolerance.

## 6. Interpretation and limitations

The observed results support the following conclusions for these
three synthetic cases:

1. The methods agree closely on the well-conditioned matrix.
2. On the matrix with condition number approximately 1e6, the global
   variance comparison understates the relative difference in small
   singular values.
3. The covariance-eigenvalue threshold removes the small artificial
   component observed in the exactly rank-deficient case.
4. The measured reconstruction errors remain on the order of 1e-15
   for these matrices and this environment.

These are observations from a single documented environment, not a
controlled comparison between BLAS vendors.

No execution times, speedups, or performance conclusions were produced.
A separate benchmark campaign must measure them.

Further experiments should investigate a broader range of condition
numbers, alternative numerical tolerances, and matrices with clustered
or nearly zero singular values. Real satellite data has not yet been
evaluated by this experiment suite.
