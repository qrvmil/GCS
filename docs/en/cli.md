# CLI reference

## Entry points

The installed command and module entry point are equivalent:

```bash
online-gcs --help
python -m online_gcs --help
```

Argument and runtime errors produce a short message and a non-zero exit code.

## Minimal run

```bash
online-gcs --scene SINGLE_SHELF --iterations 1 --keypoints 2 --prune-interval 0 --seed 42
```

| Option | Meaning |
| --- | --- |
| `--scene` | `SINGLE_SHELF`, `TWO_SHELVES`, or `TABLE_THREE_SHELVES` |
| `--iterations` | Number of online queries; must be at least 1 |
| `--keypoints` | Keypoints requested from each fallback path; must be at least 2 |
| `--seed` | Random seed used by supported planner components |
| `--prune-interval` | Region-pruning interval; `0` disables pruning |
| `--warmstart` | Build initial IRIS regions before the query loop |
| `--warmstart-seeds` | Number of warm-start seeds |
| `--smart-keypoints` | Enable the experimental keypoint selector |
| `--k-shortest-paths` | Candidate graph paths used for GCS subgraph selection |
| `--verbose` | Enable additional planner logging |

## Planner modes

Use only one baseline mode at a time:

```bash
online-gcs --scene SINGLE_SHELF --iterations 1 --rrt-only
online-gcs --scene SINGLE_SHELF --iterations 1 --opt-only
```

`--rrt-only` runs the sampling-based baseline. `--opt-only` runs trajectory
optimization of fallback paths and therefore depends on solver availability.

## Visualization and output

```bash
mkdir -p artifacts
online-gcs --scene SINGLE_SHELF --iterations 1 --output artifacts/regions.yaml
```

Add `--visualize` to print a local Meshcat URL. Region files and any other generated
data belong in ignored `results/` or `artifacts/` directories. Parallel exploration
is available with `--parallel --num-workers N`; use it only in an environment where
multiprocessing and Drake scene construction have been verified.
