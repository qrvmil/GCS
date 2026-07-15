import pytest

from online_gcs.config import OnlineGCSConfig


def test_default_config_is_valid() -> None:
    OnlineGCSConfig().validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("iterations", 0, "iterations must be at least 1"),
        ("keypoints", 1, "keypoints must be at least 2"),
        ("num_workers", 0, "num_workers must be at least 1"),
        ("animation_speed", 0.0, "animation_speed must be greater than 0"),
    ],
)
def test_invalid_numeric_config(field: str, value: float, message: str) -> None:
    config = OnlineGCSConfig(**{field: value})
    with pytest.raises(ValueError, match=message):
        config.validate()
