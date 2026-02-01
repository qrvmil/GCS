"""
GCS Path Experiments - uses pre-built IRIS regions from YAML.

Experiments:
1. Distant points - far apart configurations in C-space
2. Shelf-to-shelf - paths between shelf positions
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

from pydrake.all import (
    LoadIrisRegionsYamlFile,
    HPolyhedron,
    InverseKinematics,
    Solve,
)
from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint

from helpers.scene_builder import SceneBuilder
from helpers.gcs_panner import GCSPathPlanner
from experiments.scene_types import SceneType


@dataclass
class PathResult:
    """Result of path experiment."""
    scene: str
    num_seeds: int
    experiment_type: str  # "distant" or "shelf"
    trial: int
    path_success: bool
    path_length: float
    solve_time: float
    details: str = ""


def get_shelf_positions(scene_type: SceneType) -> List[np.ndarray]:
    """Corrected shelf positions reachable by the robot."""
    if scene_type == SceneType.SINGLE_SHELF:
        return [
            np.array([0.68, 0.0, 0.22]),   # lower
            np.array([0.68, 0.0, 0.42]),   # middle
            np.array([0.68, 0.0, 0.62]),   # upper
        ]
    elif scene_type == SceneType.TWO_SHELVES:
        return [
            np.array([0.72, -0.35, 0.25]),  # left lower
            np.array([0.72, -0.35, 0.45]),  # left middle
            np.array([0.72, -0.35, 0.65]),  # left upper
            np.array([0.72, 0.35, 0.25]),   # right lower
            np.array([0.72, 0.35, 0.45]),   # right middle
            np.array([0.72, 0.35, 0.65]),   # right upper
        ]
    else:  # TABLE_THREE_SHELVES
        z_base = 0.75 + 0.40
        return [
            np.array([0.55, 0.45, z_base - 0.15]),
            np.array([0.55, 0.45, z_base + 0.05]),
            np.array([0.55, 0.45, z_base + 0.25]),
            np.array([0.55, -0.45, z_base - 0.15]),
            np.array([0.55, -0.45, z_base + 0.05]),
            np.array([0.55, -0.45, z_base + 0.25]),
            np.array([-0.65, 0.0, z_base]),
        ]


def solve_ik(plant, plant_context, goal_pos, gripper_frame, q_initial) -> Optional[np.ndarray]:
    """Solve IK with collision avoidance."""
    plant.SetPositions(plant_context, q_initial)
    ik = InverseKinematics(plant, plant_context)
    q = ik.q()
    prog = ik.prog()
    
    prog.AddConstraint(MinimumDistanceLowerBoundConstraint(
        plant=plant, bound=0.01, plant_context=plant_context,
        influence_distance_offset=0.01
    ), q)
    
    ik.AddPositionConstraint(
        gripper_frame, np.array([0.0, 0.1, 0.0]), plant.world_frame(),
        goal_pos, goal_pos
    )
    
    prog.AddQuadraticErrorCost(np.eye(len(q_initial)), q_initial, q)
    prog.SetInitialGuess(q, q_initial)
    
    result = Solve(prog)
    return result.GetSolution(q) if result.is_success() else None


def is_collision_free(diagram, context, plant, q) -> bool:
    """Check if configuration is collision-free."""
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q)
    sg = diagram.GetSubsystemByName("scene_graph")
    sg_context = sg.GetMyContextFromRoot(context)
    query = sg.get_query_output_port().Eval(sg_context)
    return not query.HasCollisions()


def sample_collision_free(diagram, context, plant, n, rng) -> List[np.ndarray]:
    """Sample collision-free configurations."""
    q_lo = plant.GetPositionLowerLimits()
    q_hi = plant.GetPositionUpperLimits()
    configs = []
    for _ in range(n * 100):
        q = rng.uniform(q_lo, q_hi)
        if is_collision_free(diagram, context, plant, q):
            configs.append(q.copy())
            if len(configs) >= n:
                break
    return configs


class GCSExperiment:
    """GCS path experiments using pre-built regions."""
    
    def __init__(self, regions_dir: str = "gcs_results/regions",
                 output_dir: str = "gcs_experiment_results",
                 num_distant_trials: int = 5):
        self.regions_dir = Path(regions_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.num_distant_trials = num_distant_trials
        self.results: List[PathResult] = []
        
        self.scenes = [
            SceneType.SINGLE_SHELF,
            SceneType.TWO_SHELVES,
            SceneType.TABLE_THREE_SHELVES,
        ]
        self.seed_counts = [10, 20, 30, 50]
    
    def load_regions(self, scene_type: SceneType, num_seeds: int) -> List[HPolyhedron]:
        """Load pre-built regions from YAML."""
        filename = f"{scene_type.name}_{num_seeds}_1.yaml"
        filepath = self.regions_dir / filename
        if not filepath.exists():
            return []
        regions_dict = LoadIrisRegionsYamlFile(str(filepath))
        return list(regions_dict.values())
    
    def _find_distant_pair(self, configs: List[np.ndarray], 
                           regions: List[HPolyhedron],
                           min_dist: float = 2.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Find two configs in regions that are far apart."""
        # Filter configs that are in regions
        covered = [(q, i) for q in configs 
                   for i, r in enumerate(regions) if r.PointInSet(q)]
        
        if len(covered) < 2:
            return None
        
        best_pair = None
        best_dist = 0
        for i, (q1, _) in enumerate(covered):
            for q2, _ in covered[i+1:]:
                dist = np.linalg.norm(q1 - q2)
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (q1, q2)
        
        return best_pair if best_dist >= min_dist else None
    
    def run_distant_experiments(self, scene_type: SceneType, num_seeds: int,
                                 regions: List[HPolyhedron]) -> List[PathResult]:
        """Run distant point experiments."""
        results = []
        
        # Build scene
        diagram = SceneBuilder.build(scene_type)
        context = diagram.CreateDefaultContext()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyContextFromRoot(context)
        
        planner = GCSPathPlanner(regions, plant, plant_context)
        
        for trial in range(self.num_distant_trials):
            rng = np.random.default_rng(42 + trial * 1000 + num_seeds)
            
            # Sample configs
            configs = sample_collision_free(diagram, context, plant, 200, rng)
            pair = self._find_distant_pair(configs, regions)
            
            result = PathResult(
                scene=scene_type.name, num_seeds=num_seeds,
                experiment_type="distant", trial=trial+1,
                path_success=False, path_length=0.0, solve_time=0.0
            )
            
            if pair is None:
                print(f"    [distant] trial {trial+1}: No distant pair found", flush=True)
                results.append(result)
                continue
            
            q_start, q_goal = pair
            dist = np.linalg.norm(q_start - q_goal)
            
            success = planner.solve_from_configs(q_start, q_goal, build_missing_regions=False)
            result.path_success = success
            result.path_length = planner.path_length
            result.solve_time = planner.solve_time
            result.details = f"dist={dist:.2f}"
            
            status = "✓" if success else "✗"
            print(f"    [distant] trial {trial+1}: {status} len={result.path_length:.2f}", flush=True)
            results.append(result)
        
        return results
    
    def run_shelf_experiments(self, scene_type: SceneType, num_seeds: int,
                               regions: List[HPolyhedron]) -> List[PathResult]:
        """Run shelf-to-shelf experiments."""
        results = []
        
        # Build scene
        diagram = SceneBuilder.build(scene_type)
        context = diagram.CreateDefaultContext()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyContextFromRoot(context)
        
        # Get gripper frame
        try:
            wsg = plant.GetModelInstanceByName("gripper")
            gripper_frame = plant.GetFrameByName("body", wsg)
        except:
            return results
        
        planner = GCSPathPlanner(regions, plant, plant_context, gripper_frame)
        shelf_positions = get_shelf_positions(scene_type)
        
        nq = plant.num_positions()
        q_initial = np.zeros(nq)
        
        # Test all pairs
        trial = 0
        for i, pos1 in enumerate(shelf_positions):
            for j, pos2 in enumerate(shelf_positions):
                if i >= j:
                    continue
                
                trial += 1
                result = PathResult(
                    scene=scene_type.name, num_seeds=num_seeds,
                    experiment_type="shelf", trial=trial,
                    path_success=False, path_length=0.0, solve_time=0.0,
                    details=f"{i}->{j}"
                )
                
                # Solve IK
                q_start = solve_ik(plant, plant_context, pos1, gripper_frame, q_initial)
                if q_start is None:
                    results.append(result)
                    continue
                
                q_goal = solve_ik(plant, plant_context, pos2, gripper_frame, q_start)
                if q_goal is None:
                    results.append(result)
                    continue
                
                # Check coverage
                start_in = any(r.PointInSet(q_start) for r in regions)
                goal_in = any(r.PointInSet(q_goal) for r in regions)
                
                if not (start_in and goal_in):
                    results.append(result)
                    continue
                
                # Plan path
                success = planner.solve_from_configs(q_start, q_goal, build_missing_regions=False)
                result.path_success = success
                result.path_length = planner.path_length
                result.solve_time = planner.solve_time
                
                status = "✓" if success else "✗"
                print(f"    [shelf] {i}->{j}: {status} len={result.path_length:.2f}", flush=True)
                results.append(result)
                q_initial = q_goal
        
        return results
    
    def run_all(self):
        """Run all experiments."""
        total = len(self.scenes) * len(self.seed_counts)
        current = 0
        
        print("=" * 70)
        print("GCS PATH EXPERIMENTS (using pre-built regions)")
        print(f"Scenes: {[s.name for s in self.scenes]}")
        print(f"Seeds: {self.seed_counts}")
        print(f"Distant trials: {self.num_distant_trials}")
        print("=" * 70 + "\n")
        
        for scene in self.scenes:
            for num_seeds in self.seed_counts:
                current += 1
                print(f"\n[{current}/{total}] {scene.name} | seeds={num_seeds}")
                
                regions = self.load_regions(scene, num_seeds)
                if not regions:
                    print(f"  No regions found!", flush=True)
                    continue
                print(f"  Loaded {len(regions)} regions", flush=True)
                
                # Distant experiments
                print(f"  Running DISTANT experiments...", flush=True)
                distant_results = self.run_distant_experiments(scene, num_seeds, regions)
                self.results.extend(distant_results)
                d_success = sum(1 for r in distant_results if r.path_success)
                print(f"  DISTANT: {d_success}/{len(distant_results)} success", flush=True)
                
                # Shelf experiments
                print(f"  Running SHELF experiments...", flush=True)
                shelf_results = self.run_shelf_experiments(scene, num_seeds, regions)
                self.results.extend(shelf_results)
                s_success = sum(1 for r in shelf_results if r.path_success)
                print(f"  SHELF: {s_success}/{len(shelf_results)} success", flush=True)
    
    def get_statistics(self) -> pd.DataFrame:
        """Generate statistics."""
        df = pd.DataFrame([vars(r) for r in self.results])
        
        stats = df.groupby(['scene', 'num_seeds', 'experiment_type']).agg({
            'path_success': ['sum', 'count', 'mean'],
            'path_length': ['mean', 'std'],
            'solve_time': 'mean',
        }).reset_index()
        
        stats.columns = [
            'scene', 'num_seeds', 'type',
            'success_count', 'total', 'success_rate',
            'path_length_mean', 'path_length_std',
            'solve_time_mean',
        ]
        return stats
    
    def print_results(self):
        """Print formatted results."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 100)
        print("RESULTS: GCS Path Experiments")
        print("=" * 100)
        
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
                    lambda r: f"{r['path_length_mean']:.2f}±{r['path_length_std']:.2f}" 
                              if r['path_length_mean'] > 0 else "N/A",
                    axis=1
                ),
                'Solve Time': type_stats['solve_time_mean'].apply(
                    lambda x: f"{x:.3f}s" if x > 0 else "N/A"
                ),
            })
            print(display.to_string(index=False))
        
        print("\n" + "=" * 100)
    
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
    parser.add_argument("--regions", type=str, default="gcs_results/regions",
                        help="Directory with pre-built IRIS regions")
    parser.add_argument("--output", type=str, default="gcs_experiment_results",
                        help="Output directory")
    parser.add_argument("--distant-trials", type=int, default=5,
                        help="Number of distant path trials per config")
    args = parser.parse_args()
    
    experiment = GCSExperiment(
        regions_dir=args.regions,
        output_dir=args.output,
        num_distant_trials=args.distant_trials,
    )
    
    experiment.run_all()
    experiment.print_results()
    experiment.save_results()


if __name__ == "__main__":
    main()
