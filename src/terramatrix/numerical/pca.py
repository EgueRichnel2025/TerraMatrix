"""Principal component analysis using TerraMatrix BLAS and LAPACK operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from terramatrix.numerical.blas import _as_finite_matrix, dgemm, dsyrk
from terramatrix.numerical.lapack import (
    svd_factorization,
    symmetric_eigendecomposition,
)

PCAMethod = Literal["covariance", "svd"]


@dataclass(frozen=True, slots=True)
class PCAResult:
    """Results of a principal component analysis.

    Components are stored as rows and ordered by decreasing explained
    variance. Scores are the centered observations projected onto them.

    explained_variance_ratio is expressed relative to the total variance
    of the input data, not just the selected components.
    """

    mean: NDArray[np.float64]
    components: NDArray[np.float64]
    scores: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    total_variance: float
    method: PCAMethod

    @property
    def n_components(self) -> int:
        """Return the number of retained principal components."""
        return int(self.components.shape[0])


def _validate_component_count(
    value: int | None,
    maximum: int,
) -> int:
    """Validate the requested number of principal components."""
    if value is None:
        return maximum

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("'n_components' must be an integer or None.")

    count = int(value)

    if not 1 <= count <= maximum:
        raise ValueError(
            f"'n_components' must be between 1 and {maximum}, inclusive."
        )

    return count


def principal_component_analysis(
    a: ArrayLike,
    *,
    n_components: int | None = None,
    method: PCAMethod = "covariance",
) -> PCAResult:
    """Perform PCA using covariance eigendecomposition or direct SVD.

    Parameters
    ----------
    a:
        Real, finite data matrix of shape (n_samples, n_features).
        Each row is an observation and each column is a variable.
    n_components:
        Number of components to retain. Defaults to min(n_samples,
        n_features).
    method:
        "covariance" computes X_centered.T @ X_centered / (n_samples - 1)
        using BLAS DSYRK, followed by LAPACK SYEVD through the existing
        symmetric_eigendecomposition function.

        "svd" computes a direct LAPACK-backed SVD of the centered data.
        This avoids explicitly forming the covariance matrix and is
        generally preferable when numerical conditioning is a concern.

    Returns
    -------
    PCAResult
        The feature means, component directions, projected observations,
        singular values, explained variances and explained variance ratios.

    Notes
    -----
    Sample covariance uses n_samples - 1 in the denominator. Eigenvalues
    from the covariance method are clipped at zero to remove negative
    values caused by finite-precision roundoff in a positive-semidefinite
    covariance matrix.

    The two methods can produce different results for poorly conditioned
    data because forming the covariance matrix can reduce numerical accuracy.
    """
    if method not in ("covariance", "svd"):
        raise ValueError("'method' must be either 'covariance' or 'svd'.")

    matrix = _as_finite_matrix(a, "a")
    n_samples, n_features = matrix.shape

    if n_samples < 2:
        raise ValueError("PCA requires at least two observations.")

    maximum_components = min(n_samples, n_features)
    component_count = _validate_component_count(
        n_components,
        maximum_components,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(matrix, axis=0)
        centered = np.asfortranarray(matrix - mean)

    if not np.isfinite(mean).all() or not np.isfinite(centered).all():
        raise FloatingPointError(
            "Centering the data produced non-finite values."
        )

    denominator = float(n_samples - 1)

    if method == "covariance":
        covariance = dsyrk(
            centered,
            trans="T",
            alpha=1.0 / denominator,
        )
        eigenvalues, eigenvectors = symmetric_eigendecomposition(covariance)

        # EIGH returns ascending eigenvalues and stores eigenvectors in columns.
        # Reverse the order so components are ordered by decreasing variance.
        all_explained_variance = np.maximum(eigenvalues[::-1], 0.0)
        all_components = eigenvectors[:, ::-1].T.copy()

        with np.errstate(over="ignore", invalid="ignore"):
            all_singular_values = np.sqrt(
                all_explained_variance * denominator
            )

    else:
        _, all_singular_values, all_components = svd_factorization(centered)

        with np.errstate(over="ignore", invalid="ignore"):
            all_explained_variance = np.square(all_singular_values) / denominator

    if (
        not np.isfinite(all_explained_variance).all()
        or not np.isfinite(all_singular_values).all()
    ):
        raise FloatingPointError(
            "PCA variance calculations produced non-finite values."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        total_variance = float(np.sum(all_explained_variance))

    if not np.isfinite(total_variance):
        raise FloatingPointError("PCA total variance is not finite.")

    selected_components = np.asarray(
        all_components[:component_count],
        dtype=np.float64,
    )
    selected_singular_values = np.asarray(
        all_singular_values[:component_count],
        dtype=np.float64,
    )
    selected_variance = np.asarray(
        all_explained_variance[:component_count],
        dtype=np.float64,
    )

    if total_variance > 0.0:
        variance_ratio = selected_variance / total_variance
    else:
        # Constant data have zero total variance; their ratios are defined as zero.
        variance_ratio = np.zeros(component_count, dtype=np.float64)

    scores = dgemm(centered, selected_components.T)

    if not np.isfinite(variance_ratio).all():
        raise FloatingPointError(
            "PCA explained variance ratios are not finite."
        )

    return PCAResult(
        mean=np.asarray(mean, dtype=np.float64),
        components=selected_components,
        scores=scores,
        singular_values=selected_singular_values,
        explained_variance=selected_variance,
        explained_variance_ratio=np.asarray(
            variance_ratio,
            dtype=np.float64,
        ),
        total_variance=total_variance,
        method=method,
    )
