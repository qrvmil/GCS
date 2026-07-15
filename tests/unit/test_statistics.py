from types import SimpleNamespace

from online_gcs.planners.online import OnlineGCS, OnlineGCSStats


def test_statistics_use_region_counters_with_their_documented_meaning() -> None:
    planner = object.__new__(OnlineGCS)
    planner.stats = OnlineGCSStats(
        total_regions_added=3,
        total_regions_after_pruning=99,
        total_keypoints_requested=4,
        total_iris_regions_built=5,
    )
    planner.gcs_planner = SimpleNamespace(regions=[object(), object()])

    statistics = planner.get_statistics()

    assert statistics["total_regions"] == 2
    assert statistics["total_regions_added"] == 3
    assert statistics["total_regions_after_pruning"] == 2
    assert statistics["total_keypoints_requested"] == 4
    assert statistics["total_iris_regions_built"] == 5
