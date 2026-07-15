import pytest

from online_gcs import IRISRegionBuilder, SceneType


@pytest.mark.drake
def test_iris_builder_starts_without_regions() -> None:
    builder = IRISRegionBuilder(scene_type=SceneType.SINGLE_SHELF, random_seed=42)

    assert builder.num_regions == 0
    assert builder.regions == []
