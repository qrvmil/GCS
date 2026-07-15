import time

import numpy as np
import pytest

from online_gcs import SceneType
from online_gcs.exploration.parallel import ParallelExplorationCoordinator


@pytest.mark.drake
@pytest.mark.slow
def test_parallel_worker_returns_valid_region() -> None:
    coordinator = ParallelExplorationCoordinator(
        SceneType.SINGLE_SHELF,
        num_workers=1,
        random_seed=42,
    )
    rng = np.random.default_rng(42)
    seeds = [np.array([0.0, 1.04, 0.0, -1.07, 0.0, 0.34, 0.0])]
    try:
        coordinator.submit_tasks(seeds)
        deadline = time.monotonic() + 30.0
        accepted = []
        while time.monotonic() < deadline and not accepted:
            accepted = coordinator.collect_results([], rng)
            if not accepted:
                time.sleep(0.25)

        assert accepted
        assert accepted[0].PointInSet(accepted[0].ChebyshevCenter())
    finally:
        coordinator.shutdown()
