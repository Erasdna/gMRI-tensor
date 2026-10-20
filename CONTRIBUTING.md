# Contributing

1. Create a branch from `main`, e.g. `<your-name>/<what-it-does>`.
2. Set up the environment with [uv](https://docs.astral.sh/uv/) and the git hooks:

   ```bash
   uv sync
   uv run pre-commit install
   ```

3. Make the change, with `pytest` tests in `test/` for new behaviour. Plotting
   and pipeline code gets a small mock dataset (see `test/conftest.py`), so
   the tests check that it runs.
4. Before opening a pull request, these must pass:

   ```bash
   uv run pre-commit run --all-files   # black, flake8, reorder-python-imports, mypy, ...
   uv run pytest -n auto               # the fast tests, in parallel
   uv run pytest -n auto --runslow     # everything, as CI runs it
   ```

5. Open a pull request against `main`. CI runs the same checks on Linux,
   macOS and Windows with Python 3.10–3.12.
