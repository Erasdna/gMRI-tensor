"""The sandbox exposes no host files beyond the workspace. Independent of the container runtime:
configured only through the SANDBOX_* environment variables (see conftest.py)."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

KEY_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_ecdsa_sk", "id_ed25519_sk"}
KEY_SUFFIXES = {".pem", ".key"}
KEY_SEARCH_ROOTS = ["/home", "/root", "/mnt", "/media", "/srv", "/opt", "/tmp"]
SKIP_DIRS = {".vscode-server", "proc", "sys"}
HOST_PROCESSES = {"systemd", "gnome-shell", "Xorg", "Xwayland", "snapd", "gdm3"}


def _mount_points() -> set[str]:
    with open("/proc/self/mountinfo") as f:
        return {line.split()[4] for line in f}


def _is_private_key(path: Path) -> bool:
    if path.name not in KEY_NAMES and path.suffix not in KEY_SUFFIXES:
        return False
    try:
        with open(path, "rb") as f:
            return b"PRIVATE KEY" in f.read(4096)
    except OSError:
        return False


def test_host_home_contains_only_path_to_workspace(host_home: Path, workspace: Path) -> None:
    """Every directory from the host home down to the workspace holds only the next step."""
    if not workspace.is_relative_to(host_home):
        assert not host_home.exists() or not any(host_home.iterdir())
        return
    current = host_home
    for part in workspace.relative_to(host_home).parts:
        assert {p.name for p in current.iterdir()} == {part}, f"extra entries in {current}"
        current = current / part


def test_no_private_keys_outside_workspace(workspace: Path, allowed_keys: list[Path]) -> None:
    allowed = {p.resolve() for p in allowed_keys}
    found = []
    for root in KEY_SEARCH_ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and here / d != workspace]
            for name in filenames:
                path = here / name
                if path.resolve() not in allowed and _is_private_key(path):
                    found.append(path)
    assert found == []


def test_protected_paths_are_read_only(ro_paths: list[Path]) -> None:
    assert ro_paths, "SANDBOX_RO_PATHS is empty"
    for path in ro_paths:
        assert path.exists(), f"{path} is missing"
        target = path / ".sandbox_probe" if path.is_dir() else path
        mode = "x" if path.is_dir() else "a"
        try:
            with open(target, mode):
                pass
        except OSError:
            continue
        if path.is_dir():
            target.unlink()
        pytest.fail(f"{path} is writable")


def test_host_venv_is_hidden(workspace: Path) -> None:
    """The workspace .venv is a separate mount, so the host's environment is not reachable."""
    venv = workspace / ".venv"
    if venv.exists():
        assert str(venv) in _mount_points()


def test_no_git_credentials_from_host() -> None:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "", "SSH_ASKPASS": ""}
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert "password=" not in result.stdout


def test_no_container_engine_socket() -> None:
    sockets = [
        Path("/var/run/docker.sock"),
        Path("/run/docker.sock"),
        Path("/run/podman/podman.sock"),
        *Path("/run/user").glob("*/podman/podman.sock"),
        *Path("/run/user").glob("*/docker.sock"),
    ]
    assert [s for s in sockets if s.exists()] == []
    assert "DOCKER_HOST" not in os.environ


def test_no_privilege_escalation() -> None:
    if shutil.which("sudo"):
        result = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=30)
        assert result.returncode != 0
    with pytest.raises(PermissionError):
        open("/etc/shadow").close()
    with pytest.raises(PermissionError):
        tempfile.NamedTemporaryFile(dir="/usr")


def test_no_host_processes() -> None:
    names = set()
    for comm in Path("/proc").glob("[0-9]*/comm"):
        try:
            names.add(comm.read_text().strip())
        except OSError:
            pass
    assert names & HOST_PROCESSES == set()


def test_workspace_is_writable(workspace: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=workspace) as f:
        f.write(b"ok")
