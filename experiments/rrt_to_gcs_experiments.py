import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
from itertools import combinations

from online_gcs.planners.gcs import GCSPathPlanner
from online_gcs.planners.rrt_star import (
    RRTStarPlanner,
    extract_keypoints_uniform,
)
from online_gcs.regions.iris import IRISRegionBuilder
from online_gcs.scenes import SceneType, load_shelf_configurations


@dataclass
class RRTtoGCSResult:
    scene: str
    start_idx: int
    goal_idx: int
    rrt_success: bool
    rrt_path_length: float
    rrt_solve_time: float
    rrt_iterations: int
    rrt_path_points: int
    num_keypoints: int
    num_regions_built: int
    iris_build_time: float
    gcs_success: bool
    gcs_path_length: float
    gcs_solve_time: float


class RRTtoGCSExperiment:
    def __init__(
        self,
        output_dir: str = "artifacts/rrt-to-gcs",
        max_iterations: int = 10000,
        max_keypoints: int = 10,
        keypoint_epsilon: float = 0.15,
        step_size: float = 0.5,
        goal_tolerance: float = 0.15,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.regions_dir = self.output_dir / "regions"
        self.regions_dir.mkdir(parents=True, exist_ok=True)

        self.max_iterations = max_iterations
        self.max_keypoints = max_keypoints
        self.keypoint_epsilon = keypoint_epsilon
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance

        self.results: List[RRTtoGCSResult] = []

        self.scenes = [
            SceneType.SINGLE_SHELF,
            SceneType.TWO_SHELVES,
            SceneType.TABLE_THREE_SHELVES,
        ]

    def get_scene_positions(self, scene_type: SceneType) -> List[np.ndarray]:
        return load_shelf_configurations(scene_type)

    def run_single_pair(
        self,
        scene_type: SceneType,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        start_idx: int,
        goal_idx: int,
        rng_seed: int = 42,
    ) -> RRTtoGCSResult:
        result = RRTtoGCSResult(
            scene=scene_type.name,
            start_idx=start_idx,
            goal_idx=goal_idx,
            rrt_success=False,
            rrt_path_length=0.0,
            rrt_solve_time=0.0,
            rrt_iterations=0,
            rrt_path_points=0,
            num_keypoints=0,
            num_regions_built=0,
            iris_build_time=0.0,
            gcs_success=False,
            gcs_path_length=0.0,
            gcs_solve_time=0.0,
        )

        print("---[Step 1] Running BiRRT...", flush=True)
        rrt_planner = RRTStarPlanner(
            scene_type=scene_type,
            max_iterations=self.max_iterations,
            step_size=self.step_size,
            random_seed=rng_seed,
        )

        path = rrt_planner.plan_bidirectional(q_start, q_goal, goal_tolerance=self.goal_tolerance)

        if path is None:
            print("---[BiRRT] Failed, trying RRT*...", flush=True)
            rrt_planner = RRTStarPlanner(
                scene_type=scene_type,
                max_iterations=self.max_iterations,
                step_size=self.step_size,
                random_seed=rng_seed + 1000,
            )
            path = rrt_planner.plan(q_start, q_goal, goal_tolerance=self.goal_tolerance)

        result.rrt_solve_time = rrt_planner.solve_time
        result.rrt_iterations = rrt_planner.iterations_used

        if path is None:
            print("---[RRT*] Failed to find path!", flush=True)
            return result

        result.rrt_success = True
        result.rrt_path_length = rrt_planner.path_length
        result.rrt_path_points = len(path)
        print(
            f"---[RRT*] Success: length={result.rrt_path_length:.2f}, "
            f"points={len(path)}, time={result.rrt_solve_time:.2f}s",
            flush=True,
        )

        print("---[Step 2] Extracting keypoints...", flush=True)

        if len(path) <= self.max_keypoints * 2:
            keypoints = [p.copy() for p in path]
        else:
            num_keypoints = max(self.max_keypoints, len(path) // 3)
            keypoints = extract_keypoints_uniform(path, num_points=num_keypoints)

        result.num_keypoints = len(keypoints)
        print(
            f"---[Keypoints] Extracted {len(keypoints)} keypoints from {len(path)} path points",
            flush=True,
        )

        print("---[Step 3] Building IRIS regions...", flush=True)
        iris_builder = IRISRegionBuilder(scene_type, random_seed=rng_seed)
        result.iris_build_time = iris_builder.build_regions_from_seeds(keypoints)
        result.num_regions_built = iris_builder.num_regions
        print(
            f"---[IRIS] Built {result.num_regions_built} regions in {result.iris_build_time:.2f}s",
            flush=True,
        )

        if result.num_regions_built == 0:
            print("---[IRIS] No regions built, skipping GCS", flush=True)
            return result

        region_file = self.regions_dir / f"{scene_type.name}_{start_idx}_{goal_idx}.yaml"
        iris_builder.save_regions(str(region_file))

        print("---[Step 4] Running GCS planner...", flush=True)
        gcs_planner = GCSPathPlanner.from_iris_builder(iris_builder)

        t0 = time.perf_counter()
        gcs_success = gcs_planner.solve_from_configs(
            q_start,
            q_goal,
            build_missing_regions=True,
        )
        result.gcs_solve_time = time.perf_counter() - t0

        if gcs_success:
            result.gcs_success = True
            result.gcs_path_length = gcs_planner.path_length
            print(
                f"---[GCS] Success: length={result.gcs_path_length:.2f}, "
                f"time={result.gcs_solve_time:.2f}s",
                flush=True,
            )
        else:
            print("---[GCS] Failed to find path", flush=True)

        return result

    def run_scene(self, scene_type: SceneType):
        positions = self.get_scene_positions(scene_type)
        n_positions = len(positions)

        if n_positions < 2:
            print(f"---[SKIP] {scene_type.name}: not enough positions ({n_positions})")
            return

        pairs = list(combinations(range(n_positions), 2))
        n_pairs = len(pairs)

        print(f"\n{'=' * 70}")
        print(f"Scene: {scene_type.name}")
        print(f"Positions: {n_positions}, Pairs: {n_pairs}")
        print(f"{'=' * 70}")

        for pair_idx, (i, j) in enumerate(pairs):
            q_start = positions[i]
            q_goal = positions[j]

            print(f"\n  [{pair_idx + 1}/{n_pairs}] Pair ({i}, {j})")
            print(f"Start: [{', '.join(f'{x:.2f}' for x in q_start[:3])}...]")
            print(f"Goal:  [{', '.join(f'{x:.2f}' for x in q_goal[:3])}...]")

            rng_seed = 42 + pair_idx * 100 + i + j * 10

            result = self.run_single_pair(scene_type, q_start, q_goal, i, j, rng_seed)
            self.results.append(result)

            if result.rrt_success and result.gcs_success:
                improvement = (
                    (result.rrt_path_length - result.gcs_path_length) / result.rrt_path_length * 100
                )
                print(
                    f"Summary: RRT*={result.rrt_path_length:.2f} -> GCS={result.gcs_path_length:.2f} "
                    f"({improvement:+.1f}%)"
                )
            elif result.rrt_success:
                print(f"Summary: RRT* success ({result.rrt_path_length:.2f}), GCS failed")
            else:
                print("Summary: RRT* failed")

    def run_all(self):
        print("=" * 70)
        print("RRT* TO GCS PATH PLANNING EXPERIMENTS")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Max keypoints: {self.max_keypoints}")
        print(f"Step size: {self.step_size}")
        print("=" * 70)

        for scene in self.scenes:
            self.run_scene(scene)

        self.print_summary()
        self.save_results()

    def get_statistics(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(r) for r in self.results])

        stats = (
            df.groupby("scene")
            .agg(
                {
                    "rrt_success": ["sum", "count", "mean"],
                    "gcs_success": ["sum", "mean"],
                    "rrt_path_length": "mean",
                    "gcs_path_length": "mean",
                    "rrt_solve_time": "mean",
                    "gcs_solve_time": "mean",
                    "iris_build_time": "mean",
                    "num_keypoints": "mean",
                    "num_regions_built": "mean",
                }
            )
            .reset_index()
        )

        stats.columns = [
            "scene",
            "rrt_success_count",
            "total_pairs",
            "rrt_success_rate",
            "gcs_success_count",
            "gcs_success_rate",
            "rrt_path_length_mean",
            "gcs_path_length_mean",
            "rrt_time_mean",
            "gcs_time_mean",
            "iris_time_mean",
            "keypoints_mean",
            "regions_mean",
        ]

        return stats

    def print_summary(self):
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)

        stats = self.get_statistics()
        if stats.empty:
            print("No results to summarize")
            return

        for _, row in stats.iterrows():
            print(f"\n{row['scene']}:")
            print(
                f"  RRT* Success: {int(row['rrt_success_count'])}/{int(row['total_pairs'])} "
                f"({row['rrt_success_rate'] * 100:.1f}%)"
            )
            print(
                f"  GCS Success:  {int(row['gcs_success_count'])}/{int(row['total_pairs'])} "
                f"({row['gcs_success_rate'] * 100:.1f}%)"
            )

            if row["rrt_success_count"] > 0:
                print(f"  RRT* Path Length: {row['rrt_path_length_mean']:.2f}")
                print(f"  GCS Path Length:  {row['gcs_path_length_mean']:.2f}")
                print(f"  Keypoints (avg):  {row['keypoints_mean']:.1f}")
                print(f"  Regions (avg):    {row['regions_mean']:.1f}")
                print(
                    f"  Times (avg): RRT*={row['rrt_time_mean']:.1f}s, "
                    f"IRIS={row['iris_time_mean']:.1f}s, GCS={row['gcs_time_mean']:.2f}s"
                )

        df = pd.DataFrame([asdict(r) for r in self.results])
        both_success = df[(df["rrt_success"]) & (df["gcs_success"])]

        if len(both_success) > 0:
            rrt_total = both_success["rrt_path_length"].sum()
            gcs_total = both_success["gcs_path_length"].sum()
            improvement = (rrt_total - gcs_total) / rrt_total * 100
            print(f"\n Overall path length improvement (where both succeed): {improvement:+.1f}%")
            print(f"  ({len(both_success)} pairs)")

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        df = pd.DataFrame([asdict(r) for r in self.results])
        raw_file = self.output_dir / f"raw_results_{timestamp}.csv"
        df.to_csv(raw_file, index=False)

        stats = self.get_statistics()
        stats_file = self.output_dir / f"statistics_{timestamp}.csv"
        stats.to_csv(stats_file, index=False)

        print(f"\nResults saved to {self.output_dir}/")
        print(f"  Raw results: {raw_file.name}")
        print(f"  Statistics:  {stats_file.name}")


def main():
    import argparse
    import logging

    logging.getLogger("drake").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="RRT* to GCS path experiments")
    parser.add_argument(
        "--max-iterations", type=int, default=10000, help="Max RRT* iterations (default: 10000)"
    )
    parser.add_argument(
        "--max-keypoints", type=int, default=10, help="Max keypoints to extract (default: 10)"
    )
    parser.add_argument(
        "--step-size", type=float, default=0.3, help="RRT* step size in radians (default: 0.3)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.15, help="Douglas-Peucker epsilon (default: 0.15)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/rrt-to-gcs",
        help="Output directory (default: artifacts/rrt-to-gcs)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Run only specific scene (SINGLE_SHELF, TWO_SHELVES, TABLE_THREE_SHELVES)",
    )
    parser.add_argument(
        "--pair", type=str, default=None, help="Run only specific pair, format: 'i,j' (e.g., '0,1')"
    )
    args = parser.parse_args()

    experiment = RRTtoGCSExperiment(
        output_dir=args.output,
        max_iterations=args.max_iterations,
        max_keypoints=args.max_keypoints,
        keypoint_epsilon=args.epsilon,
        step_size=args.step_size,
    )

    if args.scene and args.pair:
        scene_type = SceneType[args.scene]
        i, j = map(int, args.pair.split(","))
        positions = experiment.get_scene_positions(scene_type)

        print(f"Running single pair: {args.scene} ({i}, {j})")
        result = experiment.run_single_pair(scene_type, positions[i], positions[j], i, j)
        experiment.results.append(result)
        experiment.print_summary()
        experiment.save_results()
    elif args.scene:
        scene_type = SceneType[args.scene]
        print(f"Running scene: {args.scene}")
        experiment.run_scene(scene_type)
        experiment.print_summary()
        experiment.save_results()
    else:
        experiment.run_all()


if __name__ == "__main__":
    main()
