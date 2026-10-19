#!/usr/bin/env bash
# Host: start the devcontainer, run .devcontainer/tests inside it, then check that the tests
# catch a leak (negative control: an extra host directory mounted into the container).
# Usage: bash .devcontainer/run-isolation-tests.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${CONTAINER_ENGINE:-podman}"
if command -v devcontainer > /dev/null; then
    DEVCONTAINER=(devcontainer)
else
    DEVCONTAINER=(npx -y @devcontainers/cli)
fi
PYTEST=(env SANDBOX_TESTS_REQUIRED=1 uv run pytest -q -p no:cacheprovider)

up() {
    "${DEVCONTAINER[@]}" up --docker-path "$ENGINE" --workspace-folder "$REPO" \
        --remove-existing-container "$@" > /dev/null
}
run_in() {
    "${DEVCONTAINER[@]}" exec --docker-path "$ENGINE" --workspace-folder "$REPO" "$@"
}

echo "== Sandbox tests"
up --skip-post-create
run_in uv sync --frozen --quiet
status=0
run_in "${PYTEST[@]}" .devcontainer/tests || status=1

echo "== Negative control: a mounted host directory must make the isolation tests fail"
canary="$(mktemp -d "$HOME/.sandbox-canary.XXXXXX")"
trap 'rm -rf "$canary"; up --skip-post-create || true' EXIT
up --skip-post-create --mount "type=bind,source=$canary,target=$canary"
if run_in "${PYTEST[@]}" .devcontainer/tests/test_isolation.py > /dev/null 2>&1; then
    echo "FAIL: isolation tests passed although $canary was mounted" >&2
    exit 1
fi
echo "OK: isolation tests detected the extra mount"
exit "$status"
