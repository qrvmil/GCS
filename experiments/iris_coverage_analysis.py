"""
IRIS Coverage Analysis with class-based architecture for easy GCS testing.
"""

# TODO: add .yaml with configurations setting for iris and gcs and algorithm overall

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List
from datetime import datetime

from online_gcs.regions.iris import IRISRegionBuilder
from online_gcs.scenes import SceneType


@dataclass
class ExperimentResult:
    scene: str
    num_seeds: int
    trial: int
    num_collision_free: int
    num_regions_built: int
    num_isolated: int
    iris_time: float
    isolation_rate: float
    coverage_rate: float


class IRISExperiment:
    """Run coverage experiments across multiple scenes and configurations."""

    def __init__(
        self,
        seed_counts: List[int],
        num_trials: int = 2,
        random_seed_base: int = 42,
        coverage_samples: int = 200,
        output_dir: str = "artifacts/iris-coverage",
    ):
        self.seed_counts = seed_counts
        self.num_trials = num_trials
        self.random_seed_base = random_seed_base
        self.coverage_samples = coverage_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []

    def run_single(self, scene_type: SceneType, num_seeds: int, trial: int) -> ExperimentResult:
        """Run single experiment."""
        rng_seeds = np.random.default_rng(self.random_seed_base + trial * 1000 + num_seeds)
        rng_coverage = np.random.default_rng(self.random_seed_base + trial * 2000 + num_seeds)

        print("  Building scene and IRIS regions...", flush=True)
        builder = IRISRegionBuilder(scene_type, self.random_seed_base + trial)
        iris_time = builder.build_regions(num_seeds, rng=rng_seeds)

        print(f"  Built {builder.num_regions} regions in {iris_time:.1f}s", flush=True)

        isolated = builder.count_isolated()
        print(f"  Isolated: {isolated}/{builder.num_regions}", flush=True)

        coverage = builder.estimate_coverage(self.coverage_samples, rng_coverage)
        print(f"  Coverage: {coverage * 100:.1f}%", flush=True)

        # Save regions
        regions_dir = self.output_dir / "regions"
        regions_dir.mkdir(parents=True, exist_ok=True)
        builder.save_regions(str(regions_dir / f"{scene_type.name}_{num_seeds}_{trial + 1}.yaml"))

        return ExperimentResult(
            scene=scene_type.name,
            num_seeds=num_seeds,
            trial=trial,
            num_collision_free=len(builder.seed_configs),
            num_regions_built=builder.num_regions,
            num_isolated=isolated,
            iris_time=iris_time,
            isolation_rate=builder.isolation_rate,
            coverage_rate=coverage,
        )

    def run_all(self) -> List[ExperimentResult]:
        """Run all experiments."""
        scenes = [SceneType.SINGLE_SHELF, SceneType.TWO_SHELVES, SceneType.TABLE_THREE_SHELVES]
        total = len(scenes) * len(self.seed_counts) * self.num_trials
        current = 0

        print(f"\n{'=' * 60}", flush=True)
        print(f"IRIS Coverage Analysis: {total} experiments", flush=True)
        print(f"Seeds: {self.seed_counts}, Trials: {self.num_trials}", flush=True)
        print(f"{'=' * 60}\n", flush=True)

        self.results = []
        for scene in scenes:
            for num_seeds in self.seed_counts:
                for trial in range(self.num_trials):
                    current += 1
                    print(
                        f"[{current}/{total}] {scene.name} | seeds={num_seeds} | trial={trial + 1}",
                        flush=True,
                    )
                    result = self.run_single(scene, num_seeds, trial)
                    self.results.append(result)
                    print("", flush=True)

        return self.results

    def get_statistics(self) -> pd.DataFrame:
        """Generate aggregated statistics."""
        df = pd.DataFrame([vars(r) for r in self.results])

        stats = (
            df.groupby(["scene", "num_seeds"])
            .agg(
                {
                    "num_regions_built": "mean",
                    "num_isolated": ["mean", "std"],
                    "isolation_rate": ["mean", "std"],
                    "coverage_rate": ["mean", "std"],
                    "iris_time": ["mean", "std"],
                }
            )
            .reset_index()
        )

        stats.columns = [
            "scene",
            "num_seeds",
            "avg_regions",
            "isolated_mean",
            "isolated_std",
            "isolation_rate_mean",
            "isolation_rate_std",
            "coverage_mean",
            "coverage_std",
            "iris_time_mean",
            "iris_time_std",
        ]
        return stats

    def print_results(self):
        """Print formatted results table."""
        stats = self.get_statistics()

        print("\n" + "=" * 100, flush=True)
        print("RESULTS: IRIS Coverage Analysis", flush=True)
        print("=" * 100, flush=True)

        display = pd.DataFrame(
            {
                "Scene": stats["scene"],
                "Seeds": stats["num_seeds"],
                "Regions": stats["avg_regions"].apply(lambda x: f"{x:.1f}"),
                "Isolated": stats.apply(
                    lambda r: f"{r['isolated_mean']:.1f}±{r['isolated_std']:.1f}", axis=1
                ),
                "Isolation%": stats.apply(
                    lambda r: f"{r['isolation_rate_mean'] * 100:.0f}%", axis=1
                ),
                "Coverage%": stats.apply(
                    lambda r: f"{r['coverage_mean'] * 100:.1f}%±{r['coverage_std'] * 100:.1f}%",
                    axis=1,
                ),
                "IRIS Time": stats.apply(lambda r: f"{r['iris_time_mean']:.0f}s", axis=1),
            }
        )
        print(display.to_string(index=False), flush=True)
        print("=" * 100 + "\n", flush=True)

    def save_results(self):
        """Save results to CSV files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_df = pd.DataFrame([vars(r) for r in self.results])
        raw_df.to_csv(self.output_dir / f"raw_{timestamp}.csv", index=False)

        stats = self.get_statistics()
        stats.to_csv(self.output_dir / f"stats_{timestamp}.csv", index=False)

        print(f"Results saved to {self.output_dir}/", flush=True)


def main():
    import argparse
    import logging

    logging.getLogger("drake").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="IRIS coverage analysis")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30, 50])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output", type=str, default="artifacts/iris-coverage")
    args = parser.parse_args()

    experiment = IRISExperiment(
        seed_counts=args.seeds,
        num_trials=args.trials,
        output_dir=args.output,
    )

    experiment.run_all()
    experiment.print_results()
    experiment.save_results()


if __name__ == "__main__":
    main()
