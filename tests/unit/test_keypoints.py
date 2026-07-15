import numpy as np

from online_gcs.planners.rrt_star import extract_keypoints_uniform


def test_uniform_keypoints_preserve_endpoints() -> None:
    path = [np.array([float(i), 0.0]) for i in range(5)]
    points = extract_keypoints_uniform(path, num_points=3)
    np.testing.assert_allclose(points[0], path[0])
    np.testing.assert_allclose(points[-1], path[-1])
    np.testing.assert_allclose(points[1], np.array([2.0, 0.0]))


def test_uniform_keypoints_handle_zero_length_path() -> None:
    point = np.array([1.0, 2.0])
    points = extract_keypoints_uniform([point, point.copy(), point.copy()], num_points=2)
    assert len(points) == 2
    np.testing.assert_allclose(points[0], point)
    np.testing.assert_allclose(points[-1], point)
