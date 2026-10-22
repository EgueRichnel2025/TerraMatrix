import numpy as np
import pytest

from terramatrix.numerical import dgemm


def test_dgemm_computes_rectangular_matrix_product() -> None:
    a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    b = np.array(
        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ],
        dtype=np.float64,
    )

    expected = np.array(
        [
            [58.0, 64.0],
            [139.0, 154.0],
        ],
        dtype=np.float64,
    )

    result = dgemm(a, b)

    assert result.shape == (2, 2)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)


def test_dgemm_applies_alpha_and_beta() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 0.0], [0.0, 2.0]])
    c = np.array([[5.0, 6.0], [7.0, 8.0]])
    original_c = c.copy()

    result = dgemm(a, b, alpha=2.0, beta=0.5, c=c)

    expected = 2.0 * (a @ b) + 0.5 * original_c

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(c, original_c)


def test_dgemm_rejects_incompatible_dimensions() -> None:
    a = np.ones((2, 3))
    b = np.ones((4, 2))

    with pytest.raises(ValueError, match="Incompatible matrix dimensions"):
        dgemm(a, b)


def test_dgemm_rejects_one_dimensional_inputs() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        dgemm(np.array([1.0, 2.0]), np.ones((2, 1)))


def test_dgemm_rejects_non_finite_inputs() -> None:
    a = np.array([[1.0, np.nan]])
    b = np.ones((2, 1))

    with pytest.raises(ValueError, match="finite values"):
        dgemm(a, b)


def test_dgemm_rejects_complex_inputs() -> None:
    a = np.array([[1.0 + 2.0j]])
    b = np.ones((1, 1))

    with pytest.raises(ValueError, match="real values"):
        dgemm(a, b)


def test_dgemm_requires_c_when_beta_is_nonzero() -> None:
    a = np.eye(2)
    b = np.eye(2)

    with pytest.raises(ValueError, match="'c' must be provided"):
        dgemm(a, b, beta=0.5)


def test_dgemm_rejects_invalid_output_shape() -> None:
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    c = np.ones((2, 2))

    with pytest.raises(ValueError, match="'c' must have shape"):
        dgemm(a, b, c=c)


def test_dgemm_rejects_non_finite_scalars() -> None:
    with pytest.raises(ValueError, match="'alpha' must be finite"):
        dgemm(np.eye(2), np.eye(2), alpha=float("inf"))


def test_dgemm_rejects_empty_matrices() -> None:
    a = np.empty((0, 2))
    b = np.empty((2, 1))

    with pytest.raises(ValueError, match="empty dimension"):
        dgemm(a, b)
