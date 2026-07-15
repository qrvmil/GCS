# Reproducibility

## Deterministic inputs

Use Python 3.12, record the package version, choose a packaged scene, and pass an
explicit seed. The minimal example fixes seed `42`, one query, two keypoints, and
disables pruning:

```bash
online-gcs --scene SINGLE_SHELF --iterations 1 --keypoints 2 --prune-interval 0 --seed 42
```

A seed controls supported random choices; it does not make solver timing, fetched
assets, multiprocessing order, or floating-point results identical across every
platform.

## Visualization asset

Install the optional capture dependency. In one terminal, run a fixed planner
scenario that leaves the final Meshcat scene available for inspection:

```bash
python -m pip install -e '.[demo]'
online-gcs --scene SINGLE_SHELF --seed 42 --paper-figure --figure-target-idx 1 --keypoints 10 --visualize --figure-hold-seconds 30
```

While the 30-second hold is active, copy the printed Meshcat URL into this command.
The macOS path is optional because the script auto-detects common system Chrome or
Chromium installations. If auto-detection finds nothing, omit the option only when
a Playwright Chromium browser is already installed:

```bash
python scripts/capture_demo.py --url http://localhost:7001 --output-dir artifacts/demo --width 1280 --height 720 --frames 5 --browser-executable "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

The capture tool accepts only loopback HTTP(S) URLs, removes stale numbered frame
files from the selected output directory, and leaves unrelated files untouched.
Inspect all five frames. Only if they contain genuine state changes, assemble them
with the frame-limited command printed by the script:

```bash
ffmpeg -y -framerate 2 -i artifacts/demo/frame-%02d.png -vf "scale=960:-2:flags=lanczos" -frames:v 5 -loop 0 docs/assets/online-expansion.gif
```

The repository publishes the three inspected scene screenshots, not a generated
GIF. Keep locally captured frames and GIFs under ignored paths until every frame
has been reviewed and the sequence shows real planner behavior.

## Verification

Install development and documentation extras, then run the public-tree, unit,
repository, style, and documentation checks:

```bash
python -m pip install -e '.[dev,docs]'
python scripts/check_public_tree.py
pytest tests/unit tests/repo
ruff check .
mkdocs build --strict
```

Drake integration and smoke checks exercise real scene construction:

```bash
pytest tests/integration tests/smoke -m "not slow and not visualization"
```

Slow, multiprocessing, or visualization checks should be run separately when a
change affects those paths.

## Generated output

Write generated regions, measurements, screenshots, and experiment output only
under ignored `artifacts/` or `results/` directories. Do not commit CSV files,
logs, notebooks, reports, or generated experiment results. The repository policy
test enforces this release boundary for tracked files.

## Reporting a result

Record the commit, Python and dependency versions, platform, scene, complete CLI
arguments, seed, solver availability, and whether models were already cached.
Publish the command or script needed to reproduce a claim, not only a summary
number. The repository itself intentionally contains no raw benchmark dataset.
