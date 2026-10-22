import numpy as np
import pytest

from terramatrix.numerical import dgemv


def test_dgemv_computes_matrix_vector_product() -> None:
    a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    x = np.array([1.0, 2.0, 3.0])

    result = dgemv(a, x)

    np.testing.assert_allclose(result, [14.0, 32.0])
    assert result.shape == (2,)
    assert result.dtype == np.float64


def test_dgemv_computes_transposed_product() -> None:
    a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    x = np.array([2.0, 3.0])

    result = dgemv(a, x, trans="T")

    np.testing.assert_allclose(result, [14.0, 19.0, 24.0])


def test_dgemv_applies_alpha_and_beta_without_mutating_y() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([2.0, 1.0])
    y = np.array([5.0, 6.0])
    original_y = y.copy()

    result = dgemv(a, x, alpha=2.0, beta=0.5, y=y)

    expected = 2.0 * (a @ x) + 0.5 * original_y

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(y, original_y)


def test_dgemv_rejects_invalid_transpose_option() -> None:
    with pytest.raises(ValueError, match="'trans' must be"):
        dgemv(np.eye(2), np.ones(2), trans="invalid")


def test_dgemv_rejects_incompatible_vector_length() -> None:
    with pytest.raises(ValueError, match="'x' must have length"):
        dgemv(np.ones((2, 3)), np.ones(2))


def test_dgemv_rejects_invalid_output_vector_length() -> None:
    with pytest.raises(ValueError, match="'y' must have length"):
        dgemv(np.ones((2, 3)), np.ones(3), y=np.ones(3))


def test_dgemv_requires_y_when_beta_is_nonzero() -> None:
    with pytest.raises(ValueError, match="'y' must be provided"):
        dgemv(np.eye(2), np.ones(2), beta=0.5)


def test_dgemv_rejects_non_finite_inputs() -> None:
    a = np.array([[1.0, 2.0], [3.0, np.nan]])

    with pytest.raises(ValueError, match="finite values"):
        dgemv(a, np.ones(2))


def test_dgemv_rejects_complex_vectors() -> None:
    with pytest.raises(ValueError, match="real values"):
        dgemv(np.eye(2), np.array([1.0 + 1.0j, 2.0]))


def test_dgemv_rejects_two_dimensional_vectors() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        dgemv(np.eye(2), np.ones((2, 1)))


def test_dgemv_rejects_empty_vectors() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        dgemv(np.ones((2, 1)), np.array([]))


def test_dgemv_rejects_non_finite_scalars() -> None:
    with pytest.raises(ValueError, match="'alpha' must be finite"):
        dgemv(np.eye(2), np.ones(2), alpha=float("inf"))
