from importlib import import_module
from typing import TYPE_CHECKING

from online_gcs._version import __version__

if TYPE_CHECKING:
    from online_gcs.planners.gcs import GCSPathPlanner
    from online_gcs.planners.online import OnlineGCS, OnlineGCSStats
    from online_gcs.planners.rrt_star import RRTStarPlanner
    from online_gcs.regions.iris import IRISRegionBuilder
    from online_gcs.scenes import SceneType

__all__ = [
    "GCSPathPlanner",
    "IRISRegionBuilder",
    "OnlineGCS",
    "OnlineGCSStats",
    "RRTStarPlanner",
    "SceneType",
    "__version__",
]

_LAZY_EXPORTS = {
    "GCSPathPlanner": ("online_gcs.planners.gcs", "GCSPathPlanner"),
    "IRISRegionBuilder": ("online_gcs.regions.iris", "IRISRegionBuilder"),
    "OnlineGCS": ("online_gcs.planners.online", "OnlineGCS"),
    "OnlineGCSStats": ("online_gcs.planners.online", "OnlineGCSStats"),
    "RRTStarPlanner": ("online_gcs.planners.rrt_star", "RRTStarPlanner"),
    "SceneType": ("online_gcs.scenes", "SceneType"),
}


def __getattr__(name: str) -> object:
    """Load Drake-backed public classes only when they are requested."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
