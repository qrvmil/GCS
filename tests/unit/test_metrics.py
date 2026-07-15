import numpy as np
import pytest

from online_gcs.metrics import compute_trajectory_metrics


class StraightTrajectory:
    def start_time(self) -> float:
        return 0.0

    def end_time(self) -> float:
        return 1.0

    def vector_values(self, times: np.ndarray) -> np.ndarray:
        return np.vstack([times, np.zeros_like(times), np.zeros_like(times)])


def test_straight_trajectory_metrics() -> None:
    metrics = compute_trajectory_metrics(StraightTrajectory(), num_samples=25)
    assert metrics["path_length"] == pytest.approx(1.0)
    assert metrics["curv_max"] == pytest.approx(0.0)
    assert metrics["torsion_max"] == pytest.approx(0.0)
