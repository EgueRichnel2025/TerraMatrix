"""Collection and storage of TerraMatrix environment metadata."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from terramatrix import __version__
from terramatrix.numerical import get_numerical_runtime_info


def _collect_git_metadata(project_root: Path) -> dict[str, Any]:
    """Collect Git metadata when the specified directory belongs to a repository."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": None,
            "working_tree_clean": None,
        }

    return {
        "commit": commit.stdout.strip(),
        "working_tree_clean": not status.stdout.strip(),
    }


def collect_environment_metadata(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Collect the current runtime and source-control environment.

    The reported numerical libraries are those detected in the current
    Python process. This report does not establish which library executed
    an individual operation when multiple libraries are loaded.
    """
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path.cwd().resolve()
    )

    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "project": {
            "name": "terramatrix",
            "version": __version__,
        },
        "runtime": get_numerical_runtime_info(),
        "source_control": _collect_git_metadata(root),
    }


def write_environment_metadata(
    destination: str | Path,
    project_root: str | Path | None = None,
) -> Path:
    """Write environment metadata as an indented UTF-8 JSON document."""
    output_path = Path(destination)

    metadata = collect_environment_metadata(project_root=project_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return output_path
