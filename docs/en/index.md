# Online GCS

Online GCS is a Python 3.12 research library for manipulation motion planning. Its
standard run evaluates an RRT* baseline on every query and a TrajOpt baseline when
that path exists, then checks a Graph of Convex Sets (GCS). If the current cover is
insufficient, an RRT* path supplies keypoints for new IRIS regions and a GCS retry.
Baseline runtime is therefore present even on GCS successes, and TrajOpt depends on
a compatible solver.

![Online GCS planning pipeline](../assets/pipeline.svg)

## Start here

- [Installation](installation.md) — create a supported environment and install extras.
- [Quick start](quickstart.md) — run the CLI, Python API, examples, and three scenes.
- [Architecture](architecture.md) — understand GCS, RRT*, keypoints, IRIS, and graph updates.
- [Python API](api.md) — supported public classes and generated reference.
- [CLI reference](cli.md) — commands, modes, and validation rules.
- [Reproducibility](reproducibility.md) — fixed seeds, checks, and output policy.
- [Troubleshooting](troubleshooting.md) — models, solvers, Meshcat, and runtime issues.

## Release boundary

Version `0.1.0` is a pre-1.0 research release. It supports Python 3.12 and keeps
reports, notebooks, raw experiment results, logs, and generated data outside the
public tree. See the [changelog](https://github.com/qrvmil/GCS/blob/main/CHANGELOG.md)
for release details.
