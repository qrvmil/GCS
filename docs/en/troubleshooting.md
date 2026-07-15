# Troubleshooting

## Import or command not found

Confirm Python 3.12 and install the checkout into the active environment:

```bash
python --version
python -m pip install -e .
python -c "import online_gcs; print(online_gcs.__version__)"
python -m online_gcs --help
```

If `python -m online_gcs` works but `online-gcs` does not, reactivate the virtual
environment and check that its `bin` directory is on `PATH`.

## Model downloads or scene construction

The first Drake/`manipulation` run may retrieve model assets. Check network access
and retry after the download completes. A scene-construction exception should be
reported with the scene name, platform, Drake version, and full error message.

## Slow IRIS construction

IRIS can be computationally expensive; runtime grows with scene geometry, seed
placement, and solver behavior. Start with one query and two keypoints:

```bash
online-gcs --scene SINGLE_SHELF --iterations 1 --keypoints 2 --prune-interval 0 --seed 42
```

Do not terminate a healthy process solely because its runtime differs from another
machine. Use `--verbose` when collecting a reproducible bug report.

## Solver unavailable

Trajectory optimization depends on a compatible solver in the installed Drake
environment. Try the standard online mode to distinguish an `--opt-only` solver
problem from scene construction or RRT*. Include the exact capability error in an
issue; installing a different solver may involve separate licensing terms.

## Meshcat is not visible

Run with `--visualize`, copy the printed URL into a browser on the same machine,
and keep the Python process alive. Remote, containerized, or firewalled environments
may require port forwarding that this project does not configure automatically.

## Invalid arguments

`iterations` must be at least 1, `keypoints` at least 2, pruning non-negative,
worker and path counts at least 1, and animation speed positive. Use the built-in
reference before filing an issue:

```bash
online-gcs --help
```
