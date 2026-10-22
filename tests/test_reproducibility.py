import json
from datetime import UTC, datetime
from pathlib import Path

import terramatrix.reproducibility.environment as environment_module
from terramatrix.reproducibility import (
    collect_environment_metadata,
    write_environment_metadata,
)


def test_collect_environment_metadata(monkeypatch, tmp_path: Path) -> None:
    fake_runtime = {
        "python_version": "3.12.test",
        "blas_backends": ["OpenBLAS"],
    }
    fake_git = {
        "commit": "abc123",
        "working_tree_clean": True,
    }

    monkeypatch.setattr(
        environment_module,
        "get_numerical_runtime_info",
        lambda: fake_runtime,
    )
    monkeypatch.setattr(
        environment_module,
        "_collect_git_metadata",
        lambda root: fake_git,
    )

    metadata = collect_environment_metadata(project_root=tmp_path)

    assert metadata["schema_version"] == 1
    assert metadata["project"]["name"] == "terramatrix"
    assert metadata["project"]["version"] == "0.1.0"
    assert metadata["runtime"] == fake_runtime
    assert metadata["source_control"] == fake_git

    timestamp = datetime.fromisoformat(metadata["captured_at_utc"])
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == UTC.utcoffset(timestamp)


def test_write_environment_metadata_creates_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_metadata = {
        "schema_version": 1,
        "project": {"name": "terramatrix"},
        "runtime": {"blas_backends": ["OpenBLAS"]},
        "source_control": {
            "commit": "abc123",
            "working_tree_clean": True,
        },
    }

    monkeypatch.setattr(
        environment_module,
        "collect_environment_metadata",
        lambda project_root=None: fake_metadata,
    )

    destination = tmp_path / "reports" / "environment.json"

    result = write_environment_metadata(
        destination,
        project_root=tmp_path,
    )

    assert result == destination
    assert destination.is_file()

    content = destination.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert json.loads(content) == fake_metadata


def test_git_metadata_handles_unavailable_git(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def raise_git_unavailable(*args, **kwargs):
        raise FileNotFoundError("git executable not available")

    monkeypatch.setattr(
        environment_module.subprocess,
        "run",
        raise_git_unavailable,
    )

    metadata = environment_module._collect_git_metadata(tmp_path)

    assert metadata == {
        "commit": None,
        "working_tree_clean": None,
    }
