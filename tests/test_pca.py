import numpy as np
import pytest

from terramatrix.numerical import principal_component_analysis


@pytest.mark.parametrize("method", ["covariance", "svd"])
def test_pca_returns_valid_shapes_and_centered_scores(method: str) -> None:
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal((15, 4))
    original = a.copy()

    result = principal_component_analysis(
        a,
        n_components=2,
        method=method,
    )

    assert result.method == method
    assert result.n_components == 2
    assert result.mean.shape == (4,)
    assert result.components.shape == (2, 4)
    assert result.scores.shape == (15, 2)
    assert result.singular_values.shape == (2,)
    assert result.explained_variance.shape == (2,)
    assert result.explained_variance_ratio.shape == (2,)
    assert result.total_variance > 0.0

    np.testing.assert_allclose(result.mean, np.mean(a, axis=0))
    np.testing.assert_allclose(
        result.scores,
        (a - result.mean) @ result.components.T,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.components @ result.components.T,
        np.eye(2),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize("method", ["covariance", "svd"])
def test_pca_all_components_recover_total_sample_variance(
    method: str,
) -> None:
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal((20, 4)) * np.array([1.0, 2.0, 3.0, 5.0])

    result = principal_component_analysis(a, method=method)
    expected_total_variance = float(np.sum(np.var(a, axis=0, ddof=1)))

    assert result.n_components == 4
    assert result.total_variance == pytest.approx(
        expected_total_variance,
        rel=1e-11,
        abs=1e-12,
    )
    assert np.sum(result.explained_variance_ratio) == pytest.approx(
        1.0,
        rel=1e-11,
        abs=1e-12,
    )


@pytest.mark.parametrize("method", ["covariance", "svd"])
def test_pca_reconstruction_with_all_components(method: str) -> None:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((10, 3)) + np.array([5.0, -2.0, 10.0])

    result = principal_component_analysis(a, method=method)
    reconstructed = result.scores @ result.components + result.mean

    np.testing.assert_allclose(
        reconstructed,
        a,
        rtol=1e-11,
        atol=1e-11,
    )


def test_pca_covariance_and_direct_svd_agree_on_variances() -> None:
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal((40, 5))
    a *= np.array([1.0, 2.0, 3.0, 4.0, 6.0])

    covariance_result = principal_component_analysis(a, method="covariance")
    svd_result = principal_component_analysis(a, method="svd")

    np.testing.assert_allclose(
        covariance_result.explained_variance,
        svd_result.explained_variance,
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        covariance_result.explained_variance_ratio,
        svd_result.explained_variance_ratio,
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("method", ["covariance", "svd"])
def test_pca_variance_is_nonnegative_and_descending(method: str) -> None:
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal((25, 5))

    result = principal_component_analysis(a, method=method)

    assert np.all(result.explained_variance >= 0.0)
    assert np.all(np.diff(result.explained_variance) <= 0.0)
    assert np.all(result.explained_variance_ratio >= 0.0)


@pytest.mark.parametrize("method", ["covariance", "svd"])
def test_pca_constant_data_has_zero_variance(method: str) -> None:
    a = np.full((5, 3), [2.0, -4.0, 8.0])

    result = principal_component_analysis(a, method=method)

    np.testing.assert_allclose(result.explained_variance, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        result.explained_variance_ratio,
        0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(result.scores, 0.0, atol=1e-14)
    assert result.total_variance == pytest.approx(0.0, abs=1e-14)


def test_pca_selected_components_explain_less_than_total_variance() -> None:
    rng = np.random.default_rng(20261022)
    a = rng.standard_normal((30, 5)) * np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    result = principal_component_analysis(a, n_components=2, method="svd")

    assert result.n_components == 2
    assert np.sum(result.explained_variance_ratio) < 1.0


def test_pca_requires_at_least_two_observations() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        principal_component_analysis(np.ones((1, 3)))


@pytest.mark.parametrize(
    "n_components",
    [0, -1, 1.5, True, 4],
)
def test_pca_rejects_invalid_component_count(n_components: object) -> None:
    with pytest.raises(ValueError, match="n_components"):
        principal_component_analysis(
            np.ones((5, 3)),
            n_components=n_components,
        )


def test_pca_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="'method'"):
        principal_component_analysis(np.ones((5, 3)), method="invalid")


def test_pca_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        principal_component_analysis(np.array([1.0, 2.0]))


def test_pca_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="empty dimension"):
        principal_component_analysis(np.empty((5, 0)))


def test_pca_rejects_non_finite_input() -> None:
    a = np.array([[1.0, 2.0], [3.0, np.nan], [4.0, 5.0]])

    with pytest.raises(ValueError, match="finite values"):
        principal_component_analysis(a)


def test_pca_rejects_complex_input() -> None:
    a = np.array(
        [
            [1.0 + 1.0j, 2.0],
            [3.0, 4.0],
        ]
    )

    with pytest.raises(ValueError, match="real values"):
        principal_component_analysis(a)

def test_covariance_pca_clips_roundoff_eigenvalue_for_exact_rank_deficiency() -> None:
    rng = np.random.default_rng(20261023)
    a = rng.standard_normal((30, 4))
    a[:, -1] = a[:, 0]

    covariance_result = principal_component_analysis(
        a,
        method="covariance",
    )
    direct_svd_result = principal_component_analysis(
        a,
        method="svd",
    )

    assert covariance_result.covariance_eigenvalue_tolerance is not None
    assert covariance_result.singular_values[-1] == 0.0
    assert covariance_result.explained_variance[-1] == 0.0
    assert direct_svd_result.covariance_eigenvalue_tolerance is None
    assert direct_svd_result.singular_values[-1] < 1e-12


def test_covariance_pca_preserves_resolvable_small_singular_value() -> None:
    rng = np.random.default_rng(20261024)

    left_random = rng.standard_normal((30, 4))
    left_random -= np.mean(left_random, axis=0, keepdims=True)
    left_vectors, _ = np.linalg.qr(left_random, mode="reduced")

    right_random = rng.standard_normal((4, 4))
    right_vectors, _ = np.linalg.qr(right_random)

    target_singular_values = np.array([1.0, 0.1, 0.01, 1e-6])
    a = (left_vectors * target_singular_values) @ right_vectors.T

    covariance_result = principal_component_analysis(
        a,
        method="covariance",
    )
    direct_svd_result = principal_component_analysis(
        a,
        method="svd",
    )

    assert covariance_result.singular_values[-1] > 0.0
    relative_error = abs(
        covariance_result.singular_values[-1]
        - direct_svd_result.singular_values[-1]
    ) / direct_svd_result.singular_values[-1]

    # Forming X.T @ X can amplify relative error in small singular values.
    # For this matrix, require the error to remain below 0.1%.
    assert relative_error < 1e-3
