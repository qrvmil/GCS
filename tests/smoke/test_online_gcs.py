import pytest

from online_gcs import OnlineGCS, SceneType


@pytest.mark.drake
def test_single_shelf_one_query_smoke() -> None:
    planner = OnlineGCS(
        scene_type=SceneType.SINGLE_SHELF,
        random_seed=42,
        max_iterations=1,
    )

    stats = planner.run(num_keypoints=2, prune_interval=0)

    assert stats["total_queries"] == 1
    assert stats["rrt_fallback_count"] == 1
    assert stats["gcs_success_count"] == 1
