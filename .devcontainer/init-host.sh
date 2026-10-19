#!/usr/bin/env bash
# Runs on the host before the container is created (initializeCommand), from the repo root.
set -euo pipefail

repo_name="$(basename "$PWD")"
key="$HOME/.ssh/deploy_keys/$repo_name"

# Mount targets inside the workspace must exist, or podman creates them owned by a subuid
mkdir -p .git/hooks .venv

if [ ! -f "$key" ]; then
    mkdir -p -m 700 "$(dirname "$key")"
    ssh-keygen -q -t ed25519 -N "" -C "devcontainer-$repo_name" -f "$key"
    echo "Created deploy key $key for the container. Add this public key with write access:"
    cat "$key.pub"
    origin="$(git remote get-url origin 2> /dev/null || true)"
    case "$origin" in
        git@github.com:*)
            slug="${origin#git@github.com:}"
            echo "  https://github.com/${slug%.git}/settings/keys/new"
            ;;
        *) echo "  (in the deploy key settings of: ${origin:-your remote})" ;;
    esac
fi
