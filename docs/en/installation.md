# Installation

## Requirements

- Python 3.12 (the `0.1.0` release does not claim support for other versions).
- A platform supported by the selected Drake wheel and its dependencies.
- Network access on the first run if Drake or `manipulation` needs model assets.

Create an isolated environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Install the library

An editable install is convenient for the source checkout:

```bash
python -m pip install -e .
```

Confirm that the package and CLI are visible:

```bash
python -c "import online_gcs; print(online_gcs.__version__)"
online-gcs --help
python -m online_gcs --help
```

## Optional extras

Install development and documentation tools when contributing:

```bash
python -m pip install -e '.[dev,docs]'
```

The `experiments` extra adds plotting and tabular-analysis dependencies; it is not
needed for the library, CLI, or examples.

```bash
python -m pip install -e '.[experiments]'
```

## Solver and model notes

IRIS and GCS use the solver support provided by Drake. Trajectory optimization
also depends on a compatible mathematical-program solver being available. A
missing solver is an environment capability issue, not proof that a query has no
path. Consult [Troubleshooting](troubleshooting.md) for diagnosis.
