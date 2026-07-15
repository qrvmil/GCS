import numpy as np

from online_gcs.planners.online import choose_target_index


def test_target_never_repeats_current_index() -> None:
    rng = np.random.default_rng(42)
    assert {choose_target_index(3, 0, rng) for _ in range(50)} <= {1, 2}


def test_initial_target_skips_start_configuration_zero() -> None:
    rng = np.random.default_rng(42)
    assert choose_target_index(2, 0, rng) == 1
