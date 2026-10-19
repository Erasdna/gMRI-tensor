# Claude Code devcontainer

A rootless Podman devcontainer for running Claude Code on this repo. It is adapted from the
`default` example in
[claude-devcontainer-tutorial](https://github.com/scientificcomputing/claude-devcontainer-tutorial).

**What it protects:** files on the host computer. Inside the container the model can see this
repo and nothing else from your home directory. The container is not hardened against a model
that deliberately tries to break out. Network access is not restricted.

## What the container can see

| Path | Access |
|---|---|
| The repo, at the same absolute path as on the host | read-write |
| `.git/hooks`, `.git/config`, `.devcontainer/` | read-only, because the host runs them |
| `.venv` | a container volume (the host's `.venv` is hidden) |
| `~/.ssh/deploy_keys/<repo>` on the host | read-only, at `~/.ssh/deploy_key` |
| `~/.claude`, `~/.cache` | container-only (Claude's config is lost on rebuild) |

Not visible inside: the rest of your home directory (`~/.ssh`, the host's `~/.claude`,
`~/.config`, …), the ssh-agent, VS Code's git credential helper, and the Podman/Docker socket.
The container user `vscode` has no sudo.

## Prerequisites

- Podman (rootless) and VS Code with the Dev Containers extension.
- VS Code user settings:
  ```json
  "dev.containers.dockerPath": "podman",
  "dev.containers.gitCredentialHelperConfigLocation": "none"
  ```
- Node.js, for `npx @devcontainers/cli` (only needed for `run-isolation-tests.sh`).
- On the host, install the git hook (`.git/hooks` is read-only inside the container):
  `uv run pre-commit install`.

## First start

1. In VS Code: **Dev Containers: Reopen in Container**.
2. On the first start, `init-host.sh` creates `~/.ssh/deploy_keys/<repo>` on the host and prints
   its public key. Add that key to the repo as a deploy key with **write access**
   (GitHub: Settings → Deploy keys), then rebuild the container.
3. On creation, `setup.sh` runs `uv sync`, `pre-commit run --all-files`, the project tests and
   the sandbox tests. Failures there are reported as warnings.
4. Log in to Claude Code. You need to log in again after each rebuild. Plugins (superpowers) are
   reinstalled every time VS Code attaches.

## Limitations

- `.git/config` is read-only, so `git config`, `git push -u` and `git remote add` fail inside the
  container. Use `git push origin <branch>`, and set up tracking on the host.
- Anything from the repo that you run on the host (scripts, pre-commit hooks) may have
  been edited by the model. Review the diff before you run it.
- Apptainer images are built on the host.

## Testing the sandbox

```bash
bash .devcontainer/run-isolation-tests.sh
```

The script runs `.devcontainer/tests` inside the container. It then restarts the container with
an extra host directory mounted, and checks that `test_isolation.py` fails. This replaces any
running container for this repo.

- `test_isolation.py` contains no project-specific code. It is configured only through the
  `SANDBOX_*` variables in `devcontainer.json`.
- `test_workflows.py` checks git, the Python environment and that `gMRItensor` imports.
- Outside the container the tests are skipped. When `SANDBOX_TESTS_REQUIRED=1` is set, the run
  fails instead of skipping.

## Using this in another repo

1. Copy `.devcontainer/` into the repo.
2. Edit the lines marked `PROJECT` in `devcontainer.json`, `Dockerfile`, `setup.sh` and
   `tests/test_workflows.py`: the Python version, extra apt packages and the project-specific
   tests.
3. The deploy key and the volume names come from the repo folder name, so they need no changes.
4. Add `devcontainer-lock.json` to `.gitignore`.
