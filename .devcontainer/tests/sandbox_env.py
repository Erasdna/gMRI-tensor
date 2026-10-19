"""Sandbox settings from the SANDBOX_* environment variables, as pytest fixtures and hooks."""

import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip outside the sandbox, or fail when SANDBOX_TESTS_REQUIRED=1."""
    if os.environ.get("IN_CLAUDE_DEVCONTAINER") == "1":
        return
    if os.environ.get("SANDBOX_TESTS_REQUIRED") == "1":
        raise pytest.UsageError("SANDBOX_TESTS_REQUIRED=1 but IN_CLAUDE_DEVCONTAINER is not set")
    skip = pytest.mark.skip(reason="only runs inside the sandbox container")
    for item in items:
        if HERE in item.path.parents:
            item.add_marker(skip)


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    assert value, f"{name} is not set"
    return Path(value)


def _env_paths(name: str) -> list[Path]:
    return [Path(p) for p in os.environ.get(name, "").split(":") if p]


@pytest.fixture
def workspace() -> Path:
    return _env_path("SANDBOX_WORKSPACE")


@pytest.fixture
def host_home() -> Path:
    return _env_path("SANDBOX_HOST_HOME")


@pytest.fixture
def ro_paths() -> list[Path]:
    return _env_paths("SANDBOX_RO_PATHS")


@pytest.fixture
def allowed_keys() -> list[Path]:
    return _env_paths("SANDBOX_ALLOWED_KEYS")
