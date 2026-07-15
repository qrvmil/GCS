import pytest

from online_gcs.scenes import SceneBuilder, SceneType


@pytest.mark.drake
@pytest.mark.parametrize("scene", list(SceneType))
def test_scene_builds_seven_position_plant(scene: SceneType) -> None:
    diagram = SceneBuilder.build(scene)
    plant = diagram.GetSubsystemByName("plant")
    assert plant.num_positions() == 7
