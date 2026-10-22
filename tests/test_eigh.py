import numpy as np
import pytest

from terramatrix.numerical import symmetric_eigendecomposition


@pytest.mark.parametrize("size", [1, 3, 6])
def test_eigh_reconstructs_matrix(size: int) -> None:
    rng = np.random.default_rng(20261022)
    matrix = rng.standard_normal((size, size))
    a = matrix + matrix.T

    eigenvalues, eigenvectors = symmetric_eigendecomposition(a)

    reconstructed = (eigenvectors * eigenvalues) @ eigenvectors.T

    assert eigenvalues.shape == (size,)
    assert eigenvectors.shape == (size, size)

    np.testing.assert_allclose(
        reconstructed,
        a,
        rtol=1e-12,
        atol=1e-12,
    )


def test_eigh_eigenvectors_are_orthonormal() -> None:
    a = np.array(
        [
            [4.0, 1.0, 2.0],
            [1.0, 3.0, 0.0],
            [2.0, 0.0, 2.0],
        ]
    )

    _, eigenvectors = symmetric_eigendecomposition(a)

    np.testing.assert_allclose(
        eigenvectors.T @ eigenvectors,
        np.eye(3),
        rtol=1e-12,
        atol=1e-12,
    )


def test_eigh_eigenvalues_are_ascending() -> None:
    a = np.diag([5.0, -2.0, 3.0, 1.0])

    eigenvalues, _ = symmetric_eigendecomposition(a)

    np.testing.assert_allclose(eigenvalues, [-2.0, 1.0, 3.0, 5.0])
    assert np.all(np.diff(eigenvalues) >= 0.0)


def test_eigh_handles_zero_matrix() -> None:
    a = np.zeros((3, 3))

    eigenvalues, eigenvectors = symmetric_eigendecomposition(a)

    np.testing.assert_allclose(eigenvalues, np.zeros(3))
    np.testing.assert_allclose(
        eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T,
        a,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        eigenvectors.T @ eigenvectors,
        np.eye(3),
        atol=1e-12,
    )


def test_eigh_handles_positive_semidefinite_matrix() -> None:
    b = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    a = b @ b.T

    eigenvalues, eigenvectors = symmetric_eigendecomposition(a)
    reconstructed = (eigenvectors * eigenvalues) @ eigenvectors.T

    np.testing.assert_allclose(
        reconstructed,
        a,
        rtol=1e-12,
        atol=1e-12,
    )
    assert eigenvalues[0] >= -1e-12


def test_eigh_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        symmetric_eigendecomposition(np.ones((2, 3)))


def test_eigh_rejects_non_symmetric_matrix() -> None:
    a = np.array(
        [
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="must be symmetric"):
        symmetric_eigendecomposition(a)


def test_eigh_rejects_non_finite_values() -> None:
    a = np.array(
        [
            [1.0, np.nan],
            [np.nan, 2.0],
        ]
    )

    with pytest.raises(ValueError, match="finite values"):
        symmetric_eigendecomposition(a)


def test_eigh_rejects_complex_values() -> None:
    a = np.array(
        [
            [1.0 + 1.0j, 0.0],
            [0.0, 2.0],
        ]
    )

    with pytest.raises(ValueError, match="real values"):
        symmetric_eigendecomposition(a)


def test_eigh_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        symmetric_eigendecomposition(np.array([1.0, 2.0]))


def test_eigh_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="empty dimension"):
        symmetric_eigendecomposition(np.empty((0, 0)))
