[English](README.md) | [Русский](README.ru.md)

# Online GCS

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-6f42c1.svg)](CHANGELOG.md)

Online motion planning for robot manipulation that expands a Graph of Convex Sets
only when a query needs new collision-free regions.

![Robot arm in the single-shelf scene](docs/assets/scenes/single-shelf.png)

## Overview

Online GCS combines [Drake](https://drake.mit.edu/)'s Graphs of Convex Sets (GCS),
IRIS regions, RRT*, and TrajOpt. The standard run evaluates RRT* on every query and
TrajOpt whenever that baseline finds a path, then checks the current convex-region
graph. When GCS coverage is insufficient, it grows the graph around useful points
from an RRT* path and retries GCS.

The repository provides the installable `online-gcs-planner` package, the
`online-gcs` command, a typed public Python API, deterministic examples, tests,
and matching English and Russian documentation.

## Why online GCS?

A static GCS planner needs a useful convex cover before queries arrive. Building a
large cover up front can be expensive, while a small cover may not connect a new
start and goal. This project explores a middle ground: reuse regions that already
exist and expand the graph along a successful sampling-based path only after a
query exposes a gap.

This is research software. The repository does not claim that the online strategy
outperforms every static or sampling-based planner; the included scripts and
fixed seeds are intended to make its behavior inspectable and reproducible.

## How it works

![Online GCS planning pipeline: query, RRT-star and TrajOpt baselines, GCS check, keypoints, IRIS, graph update, and GCS retry](docs/assets/pipeline.svg)

For every query, the standard run first executes an RRT* baseline and, when a path
exists, a TrajOpt baseline to collect comparative metrics. It then tries the current
GCS graph. If coverage or the GCS solve is insufficient, the fallback branch
refreshes the bidirectional RRT* path before selected keypoints seed new IRIS
regions; those regions are added to the graph and GCS is retried. Later queries can
reuse the expanded cover.

Consequently, standard-run time includes RRT* and usually TrajOpt baseline work even
when GCS succeeds. TrajOpt execution and the `--opt-only` mode depend on a compatible
solver being available; IRIS expansion adds further runtime only when it is needed.

## Quick start

Python 3.12 is required. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
online-gcs --scene SINGLE_SHELF --iterations 1 --keypoints 2 --prune-interval 0
```

The first run may download robot models used by Drake and `manipulation`. See the
[installation guide](docs/en/installation.md) for development and documentation
extras.

## Python API

The same deterministic one-query run is available through the public API:

```python
from online_gcs import OnlineGCS, SceneType

planner = OnlineGCS(
    scene_type=SceneType.SINGLE_SHELF,
    random_seed=42,
    max_iterations=1,
)
summary = planner.run(num_keypoints=2, prune_interval=0)
print(summary["total_queries"])
```

`OnlineGCS`, `OnlineGCSStats`, `GCSPathPlanner`, `RRTStarPlanner`,
`IRISRegionBuilder`, and `SceneType` form the public surface for version `0.1.0`.
See the [API reference](docs/en/api.md) for details.

## Demo and scenes

| Single shelf | Two shelves | Table with three shelves |
|:--:|:--:|:--:|
| ![Single-shelf planning scene](docs/assets/scenes/single-shelf.png) | ![Two-shelf planning scene](docs/assets/scenes/two-shelves.png) | ![Table and three-shelf planning scene](docs/assets/scenes/three-shelves.png) |
| `SINGLE_SHELF` | `TWO_SHELVES` | `TABLE_THREE_SHELVES` |

Run the deterministic non-visual and Meshcat examples with:

```bash
python examples/minimal_online.py
python examples/visualize_single_shelf.py
```

## Reproducibility

Examples use an explicit random seed. Generated regions and experiment output must
go only to ignored `artifacts/` or `results/` directories. A compact verification
run is:

```bash
python scripts/check_public_tree.py
pytest tests/unit tests/repo
ruff check .
mkdocs build --strict
```

See [Reproducibility](docs/en/reproducibility.md) for supported checks and the
boundary between examples and research experiments.

## Project status and limitations

Version `0.1.0` is a pre-1.0 research release; public interfaces may still change.
IRIS region construction can be slow, especially in harder scenes. The first
Drake/`manipulation` run may fetch model assets. Trajectory optimization depends
on a compatible solver being available in the local Drake installation. Runtime
also varies with the scene, seed, platform, and solver.

The supported release target is Python 3.12. See
[Troubleshooting](docs/en/troubleshooting.md) before reporting installation,
solver, or Meshcat issues.

## Documentation

The full documentation is available in [English](docs/en/index.md) and
[Russian](docs/ru/index.md). It covers installation, the planning concepts,
architecture, CLI and API usage, examples, and troubleshooting. The language
selector is available on every page of the built MkDocs site.

## Contributing

Focused bug fixes, tests, documentation improvements, and planning enhancements
are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md) before opening a pull request. Please use
the issue templates for reproducible bug reports and feature proposals.

## Citation

If this software supports your work, cite the metadata in
[CITATION.cff](CITATION.cff). GitHub can export it in common bibliography formats
from the repository's “Cite this repository” menu.

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Milana Krivova.
Third-party dependencies retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
