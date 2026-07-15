# Contributing

Thank you for helping improve GCS. Bug fixes, tests, documentation, and focused
planning improvements are welcome.

## Development setup

1. Fork `https://github.com/qrvmil/GCS` and clone your fork.
2. Create a branch from `main` with a descriptive name.
3. Create and activate a Python 3.12 virtual environment:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e '.[dev,docs]'
   ```

## Verification

Run the same checks used by continuous integration:

```bash
python scripts/check_public_tree.py
ruff format --check .
ruff check .
mypy src/online_gcs/config.py src/online_gcs/metrics.py src/online_gcs/cli.py
pytest tests/unit tests/repo --cov=online_gcs --cov-report=term-missing
pytest tests/integration tests/smoke -m "not slow and not visualization"
python -m build
python -m twine check dist/*
mkdocs build --strict
```

Run slow or visualization tests separately when your change affects those
paths. Add a regression test for bug fixes and update both language versions
when changing user-facing documentation.

## Pull requests

- Keep each pull request focused and explain the user-visible effect.
- Link related issues and list the commands you ran.
- Update `CHANGELOG.md` when the change affects users.
- Do not commit reports, notebooks, CSV data, logs, generated experiment
  results, caches, credentials, or personal information. Generated output
  belongs only in ignored `artifacts/` or `results/` directories.
- By contributing, you agree that your contribution is licensed under the MIT
  License in this repository.
