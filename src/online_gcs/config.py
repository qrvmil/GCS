from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OnlineGCSConfig:
    scene: str = "SINGLE_SHELF"
    iterations: int = 50
    keypoints: int = 10
    seed: int = 42
    prune_interval: int = 20
    verbose: bool = False
    visualize: bool = False
    animation_speed: float = 1.0
    smart_keypoints: bool = False
    parallel: bool = False
    num_workers: int = 2
    k_shortest_paths: int = 3
    warmstart: bool = False
    warmstart_seeds: int = 15

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.keypoints < 2:
            raise ValueError("keypoints must be at least 2")
        if self.prune_interval < 0:
            raise ValueError("prune_interval must be non-negative")
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if self.k_shortest_paths < 1:
            raise ValueError("k_shortest_paths must be at least 1")
        if self.warmstart_seeds < 1:
            raise ValueError("warmstart_seeds must be at least 1")
        if self.animation_speed <= 0:
            raise ValueError("animation_speed must be greater than 0")
