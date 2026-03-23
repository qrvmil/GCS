import sys
import time

try:
    import numpy as np
    from experiments.scene_types import SceneType
    from helpers.parallel_exploration import ParallelExplorationCoordinator
    _HAS_DRAKE = True
except ModuleNotFoundError as e:
    _HAS_DRAKE = False
    _IMPORT_ERROR = e


def main():
    if not _HAS_DRAKE:
        print("SKIP: pydrake not available:", _IMPORT_ERROR)
        return 0
    print("test_parallel_exploration: creating coordinator with 1 worker (SINGLE_SHELF)...")
    rng = np.random.default_rng(42)
    coord = ParallelExplorationCoordinator(
        scene_type=SceneType.SINGLE_SHELF,
        num_workers=1,
        random_seed=42,
    )

    seeds = [
        np.array([0.0, 1.04, 0.0, -1.07, 0.0, 0.34, 0.0], dtype=np.float64),
        np.array([0.18, 1.18, 0.06, -0.84, 0.01, 0.33, 0.0], dtype=np.float64),
    ]
    coord.submit_tasks(seeds, interval_id=0)
    print("Submitted 2 tasks, waiting for workers...")
    time.sleep(15)
    existing = []
    accepted = coord.collect_results(existing, rng)
    print(f"Collected {len(accepted)} region(s) from workers.")
    assert len(accepted) >= 1, "Expected at least one IRIS region from workers"
    for i, region in enumerate(accepted):
        center = region.ChebyshevCenter()
        assert region.PointInSet(center), f"Region {i}: Chebyshev center should be inside set"
    print("Accepted regions are valid HPolyhedra.")
    coord.shutdown()
    print("Workers shut down cleanly.")
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
