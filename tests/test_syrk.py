import numpy as np
import pytest

from terramatrix.numerical import dsyrk


def test_dsyrk_computes_column_gram_matrix() -> None:
    a = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    expected = np.array(
        [
            [35.0, 44.0],
            [44.0, 56.0],
        ]
    )

    result = dsyrk(a, trans="T")

    assert result.shape == (2, 2)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(result, result.T)


def test_dsyrk_computes_row_gram_matrix() -> None:
    a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    result = dsyrk(a, trans="N")

    np.testing.assert_allclose(result, a @ a.T)
    np.testing.assert_allclose(result, result.T)


def test_dsyrk_applies_alpha_and_beta_without_mutating_c() -> None:
    a = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    c = np.array(
        [
            [2.0, 0.5],
            [0.5, 3.0],
        ]
    )
    original_c = c.copy()

    result = dsyrk(a, trans="T", alpha=1.5, beta=0.25, c=c)
    expected = 1.5 * (a.T @ a) + 0.25 * original_c

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(c, original_c)


def test_dsyrk_rejects_invalid_transpose_option() -> None:
    with pytest.raises(ValueError, match="'trans' must be"):
        dsyrk(np.eye(2), trans="invalid")


def test_dsyrk_rejects_incompatible_c_shape() -> None:
    a = np.ones((3, 2))
    c = np.ones((3, 3))

    with pytest.raises(ValueError, match="'c' must have shape"):
        dsyrk(a, trans="T", c=c)


def test_dsyrk_rejects_non_symmetric_c() -> None:
    a = np.ones((3, 2))
    c = np.array(
        [
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="'c' must be symmetric"):
        dsyrk(a, trans="T", c=c)


def test_dsyrk_requires_c_when_beta_is_nonzero() -> None:
    with pytest.raises(ValueError, match="'c' must be provided"):
        dsyrk(np.eye(2), beta=0.5)


def test_dsyrk_rejects_non_finite_inputs() -> None:
    a = np.array(
        [
            [1.0, np.nan],
            [2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError, match="finite values"):
        dsyrk(a)


def test_dsyrk_rejects_empty_matrices() -> None:
    a = np.empty((0, 2))

    with pytest.raises(ValueError, match="empty dimension"):
        dsyrk(a)
