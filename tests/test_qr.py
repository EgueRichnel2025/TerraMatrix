import numpy as np
import pytest

from terramatrix.numerical import qr_factorization


@pytest.mark.parametrize("shape", [(6, 3), (3, 5), (4, 4)])
def test_qr_factorization_reconstructs_matrix(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal(shape)

    q, r = qr_factorization(a)

    m, n = shape
    k = min(m, n)

    assert q.shape == (m, k)
    assert r.shape == (k, n)
    assert q.dtype == np.float64
    assert r.dtype == np.float64

    np.testing.assert_allclose(q @ r, a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        q.T @ q,
        np.eye(k),
        rtol=1e-12,
        atol=1e-12,
    )


def test_qr_factorization_handles_rank_deficient_matrix() -> None:
    a = np.array(
        [
            [1.0, 2.0, 1.0],
            [3.0, 6.0, 3.0],
            [2.0, 4.0, 2.0],
            [5.0, 10.0, 5.0],
        ]
    )

    q, r = qr_factorization(a)

    np.testing.assert_allclose(q @ r, a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-12)


def test_qr_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        qr_factorization(np.array([1.0, 2.0]))


def test_qr_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="empty dimension"):
        qr_factorization(np.empty((0, 3)))


def test_qr_rejects_non_finite_values() -> None:
    a = np.array([[1.0, 2.0], [3.0, np.nan]])

    with pytest.raises(ValueError, match="finite values"):
        qr_factorization(a)


def test_qr_rejects_complex_values() -> None:
    a = np.array([[1.0 + 1.0j, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="real values"):
        qr_factorization(a)
