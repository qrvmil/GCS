# Architecture

## Planning loop

![Query to reusable-coverage pipeline](../assets/pipeline.svg)

1. A query provides start and goal configurations.
2. The standard run evaluates an RRT* baseline on every query and a TrajOpt baseline
   whenever the RRT* baseline produced a path; their metrics are recorded.
3. GCS then tries to connect the configurations through the current convex-region graph.
4. If coverage or the GCS solve is insufficient, the fallback branch refreshes the
   bidirectional RRT* path and reduces it to uniform or experimental smart keypoints.
5. IRIS grows collision-free convex regions around those seeds.
6. Non-redundant regions update the graph, and GCS is retried.
7. The expanded cover remains available to later queries in the process.

This online update is the project's research subject. It does not guarantee that
every collision-free path will be found or that expansion is faster than building
a static cover for every workload. Standard-run timing includes RRT* and usually
TrajOpt baseline evaluation even when GCS succeeds. TrajOpt needs a compatible
solver; without one, its baseline cannot provide a successful optimized trajectory.

## Components

| Module | Responsibility |
| --- | --- |
| `config.py` | Dependency-light validation of user-facing runtime options |
| `scenes.py` | Packaged scene configuration and Drake diagram construction |
| `planners/online.py` | Query loop and coordination of planning components |
| `planners/gcs.py` | Paths through a graph of convex configuration-space regions |
| `planners/rrt_star.py` | Sampling-based fallback and keypoint extraction |
| `regions/iris.py` | Collision-free IRIS region construction and persistence |
| `visualization.py` | Meshcat state, markers, paths, and animations |
| `metrics.py` | Dependency-light trajectory quality calculations |

## Graphs of Convex Sets

Each vertex represents a convex region in configuration space. Edges connect
intersecting regions. GCS optimizes a continuous trajectory through a connected
sequence of regions; a disconnected or incomplete cover can therefore prevent a
query from being solved by the current graph.

## IRIS and RRT* fallback

RRT* supplies a fallback route through free configuration space. Keypoints from
that route are seeds rather than final convex regions. IRIS inflates each seed into
a collision-free convex set, subject to the local geometry and solver behavior.
Redundancy checks limit duplicate additions before the graph is rebuilt.

## Package boundaries

The stable entry points for `0.1.0` are exported from `online_gcs`. Planner
internals, private methods, statistics dictionaries, visualization paths, and
experiment scripts may change before 1.0. CLI parsing stays dependency-light and
imports Drake-backed classes only after arguments pass validation.
