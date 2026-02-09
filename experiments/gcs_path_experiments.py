"""
GCS Path Length Experiments using existing helpers.

Experiments:
1. Distant points in C-space
2. Shelf-to-shelf paths

Uses:
- helpers.iris_region_builder.IRISRegionBuilder (build_regions_mixed)
- helpers.gcs_panner.GCSPathPlanner (solve_from_configs)
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

from helpers.iris_region_builder import IRISRegionBuilder
from helpers.gcs_panner import GCSPathPlanner
from experiments.scene_types import SceneType


@dataclass
class PathResult:
    """Single path experiment result."""
    scene: str
    num_seeds: int
    experiment_type: str  # "distant" or "shelf"
    trial: int
    success: bool
    path_length: float
    solve_time: float
    num_regions_used: int


class GCSPathExperiment:
    """GCS path length experiments."""
    
    def __init__(self, output_dir: str = "gcs_results", 
                 shelf_ratio: float = 0.3,
                 num_trials: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.regions_dir = self.output_dir / "regions"
        self.regions_dir.mkdir(exist_ok=True)
        
        self.shelf_ratio = shelf_ratio
        self.num_trials = num_trials
        self.results: List[PathResult] = []
        
        self.scenes = [
            SceneType.SINGLE_SHELF,
            SceneType.TWO_SHELVES,
            SceneType.TABLE_THREE_SHELVES,
        ]
        self.seed_counts = [10, 20, 30, 50]
    
    def _find_distant_pair(self, configs: List[np.ndarray], 
                           min_dist: float = 2.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Find two configs that are far apart."""
        best_pair = None
        best_dist = 0
        
        for i, q1 in enumerate(configs):
            for q2 in configs[i+1:]:
                dist = np.linalg.norm(q1 - q2)
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (q1, q2)
        
        if best_pair and best_dist >= min_dist:
            return best_pair
        return None
    
    def _find_configs_in_regions(self, builder: IRISRegionBuilder, 
                                  n: int, rng: np.random.Generator) -> List[np.ndarray]:
        """Sample configs that are inside IRIS regions."""
        configs = builder.sample_collision_free(n * 3, rng)
        covered = []
        for q in configs:
            for region in builder.regions:
                if region.PointInSet(q):
                    covered.append(q)
                    break
        return covered
    
    def run_single_scene(self, scene_type: SceneType, num_seeds: int, 
                         trial: int) -> Tuple[PathResult, PathResult]:
        """Run both experiments for one scene configuration."""
        rng = np.random.default_rng(42 + trial * 1000 + num_seeds)
        
        # Build IRIS regions with mixed strategy
        print(f"  Building {num_seeds} mixed IRIS regions...", flush=True)
        builder = IRISRegionBuilder(scene_type, random_seed=42 + trial)
        build_time = builder.build_regions_mixed(
            num_seeds=num_seeds, 
            shelf_ratio=self.shelf_ratio, 
            rng=rng
        )
        
        # Save regions
        regions_file = self.regions_dir / f"{scene_type.name}_{num_seeds}_{trial+1}.yaml"
        builder.save_regions(str(regions_file))
        print(f"  Saved {builder.num_regions} regions to {regions_file.name}", flush=True)
        
        # Create planner from builder
        planner = GCSPathPlanner.from_iris_builder(builder)
        
        # --- Experiment 1: Distant points ---
        print(f"  [DISTANT] Finding far apart configs...", flush=True)
        covered_configs = self._find_configs_in_regions(builder, 100, rng)
        
        distant_result = PathResult(
            scene=scene_type.name, num_seeds=num_seeds,
            experiment_type="distant", trial=trial,
            success=False, path_length=0, solve_time=0, num_regions_used=0
        )
        
        if len(covered_configs) >= 2:
            pair = self._find_distant_pair(covered_configs)
            if pair:
                q_start, q_goal = pair
                print(f"  [DISTANT] Planning path (dist={np.linalg.norm(q_start-q_goal):.2f})...", flush=True)
                
                success = planner.solve_from_configs(q_start, q_goal, build_missing_regions=False)
                distant_result.success = success
                distant_result.path_length = planner.path_length
                distant_result.solve_time = planner.solve_time
                distant_result.num_regions_used = len(planner.regions)
        
        # --- Experiment 2: Shelf-to-shelf ---
        print(f"  [SHELF] Finding shelf positions...", flush=True)
        shelf_positions = builder._get_shelf_positions()
        
        shelf_result = PathResult(
            scene=scene_type.name, num_seeds=num_seeds,
            experiment_type="shelf", trial=trial,
            success=False, path_length=0, solve_time=0, num_regions_used=0
        )
        
        if len(shelf_positions) >= 2:
            # Pick two different shelf positions
            idx1, idx2 = rng.choice(len(shelf_positions), 2, replace=False)
            pos1, pos2 = shelf_positions[idx1], shelf_positions[idx2]
            
            # Solve IK
            q_initial = np.zeros(builder.nq)
            q_start = builder._solve_ik(pos1, q_initial)
            
            if q_start is not None:
                q_goal = builder._solve_ik(pos2, q_start)
                
                if q_goal is not None:
                    # Check if both are in regions
                    start_in = any(r.PointInSet(q_start) for r in builder.regions)
                    goal_in = any(r.PointInSet(q_goal) for r in builder.regions)
                    print(f"  [SHELF] IK success. In regions: start={start_in}, goal={goal_in}", flush=True)

                    print(f"  [SHELF] Planning path...", flush=True)
                    success = planner.solve_from_configs(q_start, q_goal, build_missing_regions=True) ## !! changed to true for testing
                    shelf_result.success = success
                    shelf_result.path_length = planner.path_length
                    shelf_result.solve_time = planner.solve_time
                    shelf_result.num_regions_used = len(planner.regions)
                    
                    if start_in and goal_in:
                        print(f"  [SHELF] configs in regions", flush=True)
                    else:
                        print(f"  [SHELF] Configs not in regions", flush=True)
                else:
                    print(f"  [SHELF] IK failed for goal", flush=True)
            else:
                print(f"  [SHELF] IK failed for start", flush=True)
        
        return distant_result, shelf_result
    
    def run_all(self):
        """Run all experiments."""
        total = len(self.scenes) * len(self.seed_counts) * self.num_trials
        current = 0
        
        print("=" * 70)
        print("GCS PATH LENGTH EXPERIMENTS")
        print(f"Scenes: {[s.name for s in self.scenes]}")
        print(f"Seeds: {self.seed_counts}")
        print(f"Trials: {self.num_trials}")
        print(f"Shelf ratio: {self.shelf_ratio:.0%}")
        print(f"Total: {total} configs")
        print("=" * 70 + "\n")
        
        for scene in self.scenes:
            for num_seeds in self.seed_counts:
                for trial in range(self.num_trials):
                    current += 1
                    print(f"\n[{current}/{total}] {scene.name} | seeds={num_seeds} | trial={trial+1}")
                    
                    distant_res, shelf_res = self.run_single_scene(scene, num_seeds, trial)
                    self.results.append(distant_res)
                    self.results.append(shelf_res)
                    
                    # Print summary
                    d_status = "✓" if distant_res.success else "✗"
                    s_status = "✓" if shelf_res.success else "✗"
                    print(f"  Results: DISTANT {d_status} (len={distant_res.path_length:.2f}), "
                          f"SHELF {s_status} (len={shelf_res.path_length:.2f})")
    
    def get_statistics(self) -> pd.DataFrame:
        """Generate statistics."""
        df = pd.DataFrame([vars(r) for r in self.results])
        
        stats = df.groupby(['scene', 'num_seeds', 'experiment_type']).agg({
            'success': ['sum', 'count', 'mean'],
            'path_length': ['mean', 'std'],
            'solve_time': ['mean', 'std'],
        }).reset_index()
        
        stats.columns = [
            'scene', 'num_seeds', 'type',
            'success_count', 'total', 'success_rate',
            'path_length_mean', 'path_length_std',
            'solve_time_mean', 'solve_time_std',
        ]
        return stats
    
    def print_results(self):
        """Print formatted results."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 90)
        print("RESULTS: GCS Path Length Experiments")
        print("=" * 90)
        
        for exp_type in ['distant', 'shelf']:
            print(f"\n--- {exp_type.upper()} ---")
            type_stats = stats[stats['type'] == exp_type]
            
            display = pd.DataFrame({
                'Scene': type_stats['scene'],
                'Seeds': type_stats['num_seeds'],
                'Success': type_stats.apply(
                    lambda r: f"{int(r['success_count'])}/{int(r['total'])} ({r['success_rate']*100:.0f}%)", 
                    axis=1
                ),
                'Path Length': type_stats.apply(
                    lambda r: f"{r['path_length_mean']:.2f}±{r['path_length_std']:.2f}" if r['path_length_mean'] > 0 else "N/A",
                    axis=1
                ),
                'Solve Time': type_stats.apply(
                    lambda r: f"{r['solve_time_mean']:.3f}s" if r['solve_time_mean'] > 0 else "N/A",
                    axis=1
                ),
            })
            print(display.to_string(index=False))
        
        print("\n" + "=" * 90)
    
    def save_results(self):
        """Save results to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        raw_df = pd.DataFrame([vars(r) for r in self.results])
        raw_df.to_csv(self.output_dir / f"raw_{timestamp}.csv", index=False)
        
        stats = self.get_statistics()
        stats.to_csv(self.output_dir / f"stats_{timestamp}.csv", index=False)
        
        print(f"\nResults saved to {self.output_dir}/")


def main():
    import argparse
    import logging
    logging.getLogger("drake").setLevel(logging.WARNING)
    
    parser = argparse.ArgumentParser(description="GCS path experiments")
    parser.add_argument("--trials", type=int, default=3, help="Trials per config")
    parser.add_argument("--output", type=str, default="gcs_results", help="Output directory")
    parser.add_argument("--shelf-ratio", type=float, default=0.3, help="Ratio of shelf seeds")
    args = parser.parse_args()
    
    experiment = GCSPathExperiment(
        output_dir=args.output,
        shelf_ratio=args.shelf_ratio,
        num_trials=args.trials,
    )
    
    experiment.run_all()
    experiment.print_results()
    experiment.save_results()


if __name__ == "__main__":
    main()
