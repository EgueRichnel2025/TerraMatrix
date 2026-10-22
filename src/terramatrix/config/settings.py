"""Typed configuration loading and validation for TerraMatrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

NumericalDType = Literal["float32", "float64"]
BackendPreference = Literal["auto", "openblas", "onemkl"]


class ConfigurationError(ValueError):
    """Raised when a TerraMatrix configuration is invalid or unreadable."""


@dataclass(frozen=True, slots=True)
class NumericalSettings:
    """Configuration for numerical computations."""

    dtype: NumericalDType
    backend_preference: BackendPreference
    thread_limit: int | None


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    """Configuration for benchmark execution."""

    repetitions: int
    warmup_runs: int


@dataclass(frozen=True, slots=True)
class ReproducibilitySettings:
    """Configuration for reproducible experiments."""

    random_seed: int


@dataclass(frozen=True, slots=True)
class TerraMatrixConfig:
    """Validated configuration for the TerraMatrix application."""

    numerical: NumericalSettings
    benchmark: BenchmarkSettings
    reproducibility: ReproducibilitySettings


def _require_mapping(value: object, section: str) -> dict[str, object]:
    """Require a string-keyed mapping for a configuration section."""
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Configuration section '{section}' must be a mapping."
        )

    if any(not isinstance(key, str) for key in value):
        raise ConfigurationError(
            f"Configuration section '{section}' must contain string keys."
        )

    return value


def _reject_unknown_keys(
    values: dict[str, object],
    allowed: set[str],
    section: str,
) -> None:
    """Reject unsupported keys instead of silently ignoring them."""
    unknown = sorted(key for key in values if key not in allowed)

    if unknown:
        names = ", ".join(unknown)
        raise ConfigurationError(
            f"Unknown configuration key(s) in '{section}': {names}."
        )


def _read_choice(
    value: object,
    field: str,
    choices: tuple[str, ...],
) -> str:
    """Validate a string against the supported values."""
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(choices)
        raise ConfigurationError(
            f"'{field}' must be one of: {allowed}."
        )

    return value


def _read_integer(
    value: object,
    field: str,
    *,
    minimum: int,
) -> int:
    """Validate an integer while explicitly rejecting booleans."""
    if type(value) is not int or value < minimum:
        raise ConfigurationError(
            f"'{field}' must be an integer greater than or equal to {minimum}."
        )

    return value


def _read_optional_positive_integer(
    value: object,
    field: str,
) -> int | None:
    """Validate a positive integer or None."""
    if value is None:
        return None

    return _read_integer(value, field, minimum=1)


def load_config(path: str | Path) -> TerraMatrixConfig:
    """Load and validate a TerraMatrix YAML configuration file.

    Missing sections and settings receive documented defaults. Unknown
    keys are rejected to catch spelling errors and unsupported options.

    The returned configuration describes requested parameters. It does
    not itself change the loaded BLAS backend or enforce a thread limit.
    """
    config_path = Path(path)

    try:
        content = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(
            f"Unable to read configuration file '{config_path}': {exc}"
        ) from exc

    try:
        raw_config = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in '{config_path}': {exc}"
        ) from exc

    if raw_config is None:
        raise ConfigurationError(
            f"Configuration file '{config_path}' is empty."
        )

    root = _require_mapping(raw_config, "root")
    _reject_unknown_keys(
        root,
        {"numerical", "benchmark", "reproducibility"},
        "root",
    )

    numerical = _require_mapping(root.get("numerical", {}), "numerical")
    benchmark = _require_mapping(root.get("benchmark", {}), "benchmark")
    reproducibility = _require_mapping(
        root.get("reproducibility", {}),
        "reproducibility",
    )

    _reject_unknown_keys(
        numerical,
        {"dtype", "backend_preference", "thread_limit"},
        "numerical",
    )
    _reject_unknown_keys(
        benchmark,
        {"repetitions", "warmup_runs"},
        "benchmark",
    )
    _reject_unknown_keys(
        reproducibility,
        {"random_seed"},
        "reproducibility",
    )

    dtype = _read_choice(
        numerical.get("dtype", "float64"),
        "numerical.dtype",
        ("float32", "float64"),
    )
    backend_preference = _read_choice(
        numerical.get("backend_preference", "auto"),
        "numerical.backend_preference",
        ("auto", "openblas", "onemkl"),
    )

    thread_limit = _read_optional_positive_integer(
        numerical.get("thread_limit"),
        "numerical.thread_limit",
    )

    repetitions = _read_integer(
        benchmark.get("repetitions", 5),
        "benchmark.repetitions",
        minimum=1,
    )
    warmup_runs = _read_integer(
        benchmark.get("warmup_runs", 1),
        "benchmark.warmup_runs",
        minimum=0,
    )
    random_seed = _read_integer(
        reproducibility.get("random_seed", 42),
        "reproducibility.random_seed",
        minimum=0,
    )

    return TerraMatrixConfig(
        numerical=NumericalSettings(
            dtype=cast(NumericalDType, dtype),
            backend_preference=cast(
                BackendPreference,
                backend_preference,
            ),
            thread_limit=thread_limit,
        ),
        benchmark=BenchmarkSettings(
            repetitions=repetitions,
            warmup_runs=warmup_runs,
        ),
        reproducibility=ReproducibilitySettings(
            random_seed=random_seed,
        ),
    )
