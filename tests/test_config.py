from pathlib import Path

import pytest

from terramatrix.config import ConfigurationError, load_config

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "config" / "default.yaml"
)


def test_default_configuration_is_valid() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.numerical.dtype == "float64"
    assert config.numerical.backend_preference == "auto"
    assert config.numerical.thread_limit is None

    assert config.benchmark.repetitions == 5
    assert config.benchmark.warmup_runs == 1

    assert config.reproducibility.random_seed == 42


def test_missing_settings_receive_defaults(tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("numerical: {}\n", encoding="utf-8")

    config = load_config(path)

    assert config.numerical.dtype == "float64"
    assert config.numerical.backend_preference == "auto"
    assert config.benchmark.repetitions == 5
    assert config.reproducibility.random_seed == 42


@pytest.mark.parametrize(
    ("content", "expected_message"),
    [
        (
            "unexpected: true\n",
            "Unknown configuration key(s) in 'root'",
        ),
        (
            "numerical:\n  unknown_option: 3\n",
            "Unknown configuration key(s) in 'numerical'",
        ),
    ],
)
def test_unknown_configuration_keys_are_rejected(
    tmp_path: Path,
    content: str,
    expected_message: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(path)

    assert expected_message in str(exc_info.value)


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text("numerical: [\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_config(path)


def test_root_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- numerical\n- benchmark\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="root.*mapping"):
        load_config(path)


@pytest.mark.parametrize(
    ("content", "expected_message"),
    [
        (
            "numerical:\n  dtype: float16\n",
            "'numerical.dtype'",
        ),
        (
            "numerical:\n  backend_preference: vendor\n",
            "'numerical.backend_preference'",
        ),
        (
            "numerical:\n  thread_limit: 0\n",
            "'numerical.thread_limit'",
        ),
        (
            "numerical:\n  thread_limit: true\n",
            "'numerical.thread_limit'",
        ),
        (
            "benchmark:\n  repetitions: 0\n",
            "'benchmark.repetitions'",
        ),
        (
            "benchmark:\n  warmup_runs: -1\n",
            "'benchmark.warmup_runs'",
        ),
        (
            "reproducibility:\n  random_seed: -1\n",
            "'reproducibility.random_seed'",
        ),
    ],
)
def test_invalid_settings_are_rejected(
    tmp_path: Path,
    content: str,
    expected_message: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(path)

    assert expected_message in str(exc_info.value)


def test_missing_configuration_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigurationError, match="Unable to read configuration"):
        load_config(path)
