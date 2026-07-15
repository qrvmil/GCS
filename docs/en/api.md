# Python API

## Public imports

The package root exposes the supported classes without importing Drake until a
Drake-backed object is requested:

```python
from online_gcs import (
    GCSPathPlanner,
    IRISRegionBuilder,
    OnlineGCS,
    OnlineGCSStats,
    RRTStarPlanner,
    SceneType,
)
```

`OnlineGCS` is the high-level interface. The lower-level planners and region
builder are available for research code that needs explicit control, but their
method-level details may evolve before 1.0.

## Online planner

::: online_gcs.OnlineGCS
    options:
      members:
        - run
        - get_statistics
        - save_regions

## Statistics

::: online_gcs.OnlineGCSStats

## GCS planner

::: online_gcs.GCSPathPlanner
    options:
      members:
        - from_iris_builder
        - from_yaml
        - solve_from_configs

## RRT* planner

::: online_gcs.RRTStarPlanner
    options:
      members:
        - plan
        - plan_bidirectional

## IRIS region builder

::: online_gcs.IRISRegionBuilder
    options:
      members:
        - build_regions
        - build_regions_from_seeds
        - save_regions
        - load_regions

## Scenes

::: online_gcs.SceneType
