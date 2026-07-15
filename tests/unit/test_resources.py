import numpy as np

from online_gcs.scenes import SceneType, load_shelf_configurations


def test_each_scene_has_seven_dof_targets() -> None:
    for scene in SceneType:
        targets = load_shelf_configurations(scene)
        assert len(targets) >= 2
        assert all(isinstance(target, np.ndarray) and target.shape == (7,) for target in targets)


def test_resource_loader_returns_fresh_arrays() -> None:
    first = load_shelf_configurations(SceneType.SINGLE_SHELF)
    second = load_shelf_configurations(SceneType.SINGLE_SHELF)
    first[0][0] = 999.0
    assert second[0][0] != 999.0
