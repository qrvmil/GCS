import pytest

from online_gcs import OnlineGCS, SceneType


def test_zero_max_iterations_is_rejected_before_scene_construction() -> None:
    with pytest.raises(ValueError, match="max_iterations must be at least 1"):
        OnlineGCS(SceneType.SINGLE_SHELF, max_iterations=0)
