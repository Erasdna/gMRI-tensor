"""What still has to work inside the sandbox. Tests marked PROJECT are specific to this repo."""

import subprocess
import sys
from pathlib import Path


def _run(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)


def test_git_status(workspace: Path) -> None:
    assert _run(["git", "status", "--short"], workspace).returncode == 0


def test_git_remote_reachable(workspace: Path) -> None:
    """Uses the deploy key from GIT_SSH_COMMAND; the host ssh-agent is not used."""
    result = _run(["git", "ls-remote", "origin", "HEAD"], workspace)
    assert result.returncode == 0, result.stderr


def test_python_is_project_venv(workspace: Path) -> None:
    assert Path(sys.prefix) == workspace / ".venv"


def test_project_importable(workspace: Path) -> None:  # PROJECT
    result = _run([sys.executable, "-c", "import gMRItensor"], workspace)
    assert result.returncode == 0, result.stderr
