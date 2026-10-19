#!/usr/bin/env bash
# postCreateCommand: install the project and run the local CI checks. Failures only warn.
set -uo pipefail

warn() { echo "WARNING: $*" >&2; }

uv sync --frozen || warn "uv sync failed"
uv run pre-commit run --all-files || warn "pre-commit reported problems"
uv run pytest -q || warn "project tests failed"
SANDBOX_TESTS_REQUIRED=1 uv run pytest -q -p no:cacheprovider .devcontainer/tests \
    || warn "devcontainer isolation/workflow tests failed"

if [ ! -x .git/hooks/pre-commit ]; then
    echo "Install the git hook on the host (.git/hooks is read-only here): uv run pre-commit install"
fi
echo "Container setup complete."
