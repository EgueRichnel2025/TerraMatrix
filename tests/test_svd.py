import numpy as np
import pytest

from terramatrix.numerical import svd_factorization


@pytest.mark.parametrize("shape", [(6, 3), (3, 5), (4, 4)])
def test_svd_reconstructs_matrix(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal(shape)

    u, singular_values, vt = svd_factorization(a)

    m, n = shape
    k = min(m, n)

    assert u.shape == (m, k)
    assert singular_values.shape == (k,)
    assert vt.shape == (k, n)

    assert u.dtype == np.float64
    assert singular_values.dtype == np.float64
    assert vt.dtype == np.float64

    reconstructed = (u * singular_values) @ vt

    np.testing.assert_allclose(
        reconstructed,
        a,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("shape", [(6, 3), (3, 5), (4, 4)])
def test_svd_singular_vectors_are_orthonormal(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal(shape)

    u, _, vt = svd_factorization(a)
    k = min(shape)

    np.testing.assert_allclose(
        u.T @ u,
        np.eye(k),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        vt @ vt.T,
        np.eye(k),
        rtol=1e-12,
        atol=1e-12,
    )


def test_svd_values_are_nonnegative_and_descending() -> None:
    a = np.array(
        [
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )

    _, singular_values, _ = svd_factorization(a)

    assert np.all(singular_values >= 0.0)
    assert np.all(np.diff(singular_values) <= 0.0)


def test_svd_reconstructs_rank_deficient_matrix() -> None:
    a = np.array(
        [
            [1.0, 2.0, 1.0],
            [3.0, 6.0, 3.0],
            [2.0, 4.0, 2.0],
            [5.0, 10.0, 5.0],
        ]
    )

    u, singular_values, vt = svd_factorization(a)

    reconstructed = (u * singular_values) @ vt

    np.testing.assert_allclose(
        reconstructed,
        a,
        rtol=1e-12,
        atol=1e-12,
    )


def test_svd_handles_zero_matrix() -> None:
    a = np.zeros((3, 2))

    u, singular_values, vt = svd_factorization(a)

    np.testing.assert_allclose(
        (u * singular_values) @ vt,
        a,
        atol=1e-14,
    )
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(vt @ vt.T, np.eye(2), atol=1e-12)


def test_svd_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        svd_factorization(np.array([1.0, 2.0]))


def test_svd_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="empty dimension"):
        svd_factorization(np.empty((0, 3)))


def test_svd_rejects_non_finite_values() -> None:
    a = np.array(
        [
            [1.0, 2.0],
            [3.0, np.nan],
        ]
    )

    with pytest.raises(ValueError, match="finite values"):
        svd_factorization(a)


def test_svd_rejects_complex_values() -> None:
    a = np.array(
        [
            [1.0 + 1.0j, 2.0],
            [3.0, 4.0],
        ]
    )

    with pytest.raises(ValueError, match="real values"):
        svd_factorization(a)
