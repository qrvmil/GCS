from dataclasses import dataclass
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np
from pydrake.all import (
    HPolyhedron,
    StartMeshcat,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    Rgba,
    Sphere,
    Cylinder,
)
from pydrake.visualization import AddDefaultVisualization

from manipulation.scenarios import AddIiwa, AddWsg
from manipulation.utils import ConfigureParser

from experiments.scene_types import SceneType
from helpers.gcs_panner import GCSPathPlanner
from helpers.iris_region_builder import IRISRegionBuilder
from helpers.rrt_star_planner import RRTStarPlanner
from helpers.rrt_star_planner import extract_keypoints_uniform

# Algorithm:
# 1. Receive target configurations sequentially
# 2. For each target, if both start AND goal are inside GCS regions, plan with GCS
# 3. If not in GCS, run RRT* and build IRIS regions from the path keypoints
# 4. Add new regions to GCS graph, gradually expanding coverage

# TODO: add visualization
# TODO: add logging

@dataclass
class OnlineGCSStats:
    total_regions: int
    gcs_success_count: int
    rrt_fallback_count: int
    total_queries: int
    gcs_success_rate: float
    total_regions_added: int
    total_regions_after_pruning: int
    stats_per_each_query: Dict[str, List[float]]


class OnlineGCS:
    def __init__(self, scene_type: SceneType, random_seed: int = 42,
                 logging: bool = False, visualize: bool = False,
                 max_iterations: int = 100):
        self.scene_type = scene_type
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.logging = logging
        self.visualize = visualize
        self.max_iterations = max_iterations

        self.iris_region_builder = IRISRegionBuilder(scene_type=scene_type, random_seed=random_seed)
        
        self.gcs_planner = GCSPathPlanner.from_iris_builder(self.iris_region_builder)
        
        self.rrt_planner = RRTStarPlanner(
            scene_type=scene_type,
            random_seed=random_seed,
        )
        
        self.scene_to_config_key = {
            SceneType.SINGLE_SHELF: "positions_single_shelf_cspace",
            SceneType.TWO_SHELVES: "positions_two_shelves_cspace",
            SceneType.TABLE_THREE_SHELVES: "positions_three_shelves_cspace",
        }
        
        config_path = Path(__file__).parent.parent / "experiments" / "configs" / "shelf_cspace_positions.yaml"
        self.shelf_configs = self._load_cspace_positions(str(config_path))
        self.shelf_configs = self.shelf_configs[self.scene_to_config_key[scene_type]]
        
        self.current_qpos = np.array(self.shelf_configs[0]) if self.shelf_configs else np.zeros(7)

        self.stats = OnlineGCSStats()
        
        self.meshcat = None
        self.vis_diagram = None
        self.vis_context = None
        self.vis_plant = None
        self.vis_plant_context = None
        self.target_point_count = 0
        self.path_line_count = 0
        
        if visualize:
            self._setup_visualization()



    def check_if_point_in_iris_regions(self, point: np.ndarray) -> bool:
        """Check if a configuration point is inside any existing IRIS region."""
        if point is None:
            return False
        return self.gcs_planner._find_containing_region(point) != -1
    
    # ==================== VISUALIZATION METHODS ====================
    
    def _setup_visualization(self):
        """Setup Meshcat visualization with a new diagram."""
        print("Setting up Meshcat visualization...", flush=True)
        
        self.meshcat = StartMeshcat()
        
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        
        if self.scene_type == SceneType.SINGLE_SHELF:
            self._build_single_shelf_scene(plant)
        elif self.scene_type == SceneType.TWO_SHELVES:
            self._build_two_shelves_scene(plant)
        elif self.scene_type == SceneType.TABLE_THREE_SHELVES:
            self._build_table_three_shelves_scene(plant)
        
        plant.Finalize()
        
        AddDefaultVisualization(builder, self.meshcat)
        
        self.vis_diagram = builder.Build()
        self.vis_context = self.vis_diagram.CreateDefaultContext()
        self.vis_plant = self.vis_diagram.GetSubsystemByName("plant")
        self.vis_plant_context = self.vis_plant.GetMyContextFromRoot(self.vis_context)
        
        self.vis_plant.SetPositions(self.vis_plant_context, self.current_qpos)
        self.vis_diagram.ForcedPublish(self.vis_context)
        
        print(f"Meshcat URL: {self.meshcat.web_url()}", flush=True)
    
    def _build_single_shelf_scene(self, plant):
        """Build single shelf scene for visualization."""
        iiwa = AddIiwa(plant)
        AddWsg(plant, iiwa, welded=True, sphere=False)
        parser = Parser(plant)
        ConfigureParser(parser)
        shelf = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("shelves_body", shelf),
            RigidTransform([0.88, 0, 0.4]),
        )
    
    def _build_two_shelves_scene(self, plant):
        """Build two shelves scene for visualization."""
        iiwa = AddIiwa(plant)
        AddWsg(plant, iiwa, welded=True, sphere=False)
        parser = Parser(plant)
        ConfigureParser(parser)
        
        s1 = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
        plant.RenameModelInstance(s1, "shelves1")
        s2 = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
        plant.RenameModelInstance(s2, "shelves2")
        
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("shelves_body", s1),
                         RigidTransform([0.95, -0.35, 0.40]))
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("shelves_body", s2),
                         RigidTransform([0.95, 0.35, 0.40]))
    
    def _build_table_three_shelves_scene(self, plant):
        """Build table with three shelves scene for visualization."""
        from pydrake.all import SpatialInertia, UnitInertia, CoulombFriction, Box
        
        z_table = 0.75
        lx, ly, thickness = 2.3, 1.8, 0.10
        inertia = SpatialInertia(mass=30.0, p_PScm_E=[0, 0, 0],
                                  G_SP_E=UnitInertia.SolidBox(lx, ly, thickness))
        table = plant.AddRigidBody("table", inertia)
        plant.WeldFrames(plant.world_frame(), table.body_frame(),
                         RigidTransform([0, 0, z_table - 0.5 * thickness]))
        friction = CoulombFriction(0.9, 0.8)
        plant.RegisterCollisionGeometry(table, RigidTransform(), Box(lx, ly, thickness),
                                        "table_col", friction)
        plant.RegisterVisualGeometry(table, RigidTransform(), Box(lx, ly, thickness),
                                     "table_vis", np.array([0.55, 0.25, 0.10, 1.0]))
        
        parser = Parser(plant)
        ConfigureParser(parser)
        
        iiwa = parser.AddModelsFromUrl(
            "package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf")[0]
        plant.RenameModelInstance(iiwa, "iiwa")
        plant.WeldFrames(plant.world_frame(),
                         plant.GetBodyByName("iiwa_link_0", iiwa).body_frame(),
                         RigidTransform([0, 0, z_table]))
        AddWsg(plant, iiwa, welded=True, sphere=False)
        
        z_sh = z_table + 0.40
        for name, xyz, yaw in [("sl", [0.85, 0.65, z_sh], 45),
                               ("sr", [0.85, -0.65, z_sh], -45),
                               ("sb", [-0.95, 0.0, z_sh], 180)]:
            m = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
            plant.RenameModelInstance(m, name)
            plant.WeldFrames(plant.world_frame(),
                             plant.GetBodyByName("shelves_body", m).body_frame(),
                             RigidTransform(RollPitchYaw(0, 0, np.deg2rad(yaw)), xyz))
    
    def _update_robot_visualization(self, q: np.ndarray):
        """Update robot position in visualization."""
        if self.meshcat is None:
            return
        self.vis_plant.SetPositions(self.vis_plant_context, q)
        self.vis_diagram.ForcedPublish(self.vis_context)
    
    def _get_ee_position(self, q: np.ndarray) -> np.ndarray:
        """Get end-effector position for a given configuration."""
        self.vis_plant.SetPositions(self.vis_plant_context, q)
        try:
            wsg = self.vis_plant.GetModelInstanceByName("gripper")
            gripper_frame = self.vis_plant.GetFrameByName("body", wsg)
            return self.vis_plant.CalcPointsPositions(
                self.vis_plant_context,
                gripper_frame,
                np.array([[0], [0.1], [0]]),
                self.vis_plant.world_frame()
            ).flatten()
        except Exception:
            return np.zeros(3)
    
    def _visualize_target_point(self, q_target: np.ndarray, color: Rgba = None):
        """Visualize target configuration as a sphere at end-effector position."""
        if self.meshcat is None:
            return
        
        ee_pos = self._get_ee_position(q_target)
        
        if color is None:
            hue = (self.target_point_count * 0.618033988749895) % 1.0
            color = self._hsv_to_rgba(hue, 0.8, 0.9, 0.8)
        
        point_name = f"targets/target_{self.target_point_count}"
        self.meshcat.SetObject(point_name, Sphere(0.025), color)
        self.meshcat.SetTransform(point_name, RigidTransform(ee_pos))
        
        self.target_point_count += 1
    
    def _visualize_path(self, path: List[np.ndarray], color: Rgba = None, 
                        use_gcs: bool = True):
        """Visualize a path as connected line segments."""
        if self.meshcat is None or len(path) < 2:
            return
        
        if color is None:
            color = Rgba(0.2, 0.8, 0.2, 0.7) if use_gcs else Rgba(0.9, 0.5, 0.1, 0.7)
        
        for i in range(len(path) - 1):
            pos1 = self._get_ee_position(path[i])
            pos2 = self._get_ee_position(path[i + 1])
            
            self._draw_line_segment(pos1, pos2, color, 
                                   f"paths/path_{self.path_line_count}/seg_{i}")
        
        self.path_line_count += 1
    
    def _draw_line_segment(self, p1: np.ndarray, p2: np.ndarray, 
                           color: Rgba, name: str):
        """Draw a line segment between two 3D points as a thin cylinder."""
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            return
        
        midpoint = (p1 + p2) / 2
        
        direction_normalized = direction / length
        
        z_axis = np.array([0, 0, 1])
        
        if np.abs(np.dot(direction_normalized, z_axis)) > 0.999:
            rotation = RollPitchYaw(0, 0, 0)
        else:
            axis = np.cross(z_axis, direction_normalized)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
            
            rotation = RollPitchYaw(0, np.arctan2(
                np.sqrt(direction_normalized[0]**2 + direction_normalized[1]**2),
                direction_normalized[2]
            ), np.arctan2(direction_normalized[1], direction_normalized[0]))
        
        self.meshcat.SetObject(name, Cylinder(0.003, length), color)
        self.meshcat.SetTransform(name, RigidTransform(rotation, midpoint))
    
    def _animate_trajectory(self, trajectory, duration: float = 2.0, 
                           num_samples: int = 50):
        """Animate robot along a GCS trajectory."""
        if self.meshcat is None or trajectory is None:
            return
        
        times = np.linspace(trajectory.start_time(), trajectory.end_time(), num_samples)
        
        for t in times:
            q = trajectory.value(t).flatten()
            self._update_robot_visualization(q)
            time.sleep(duration / num_samples)
    
    def _animate_path(self, path: List[np.ndarray], duration: float = 2.0):
        """Animate robot along a path (list of configurations)."""
        if self.meshcat is None or not path:
            return
        
        dt = duration / len(path)
        for q in path:
            self._update_robot_visualization(q)
            time.sleep(dt)
    
    def _hsv_to_rgba(self, h: float, s: float, v: float, a: float = 1.0) -> Rgba:
        """Convert HSV to RGBA color."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return Rgba(r, g, b, a)
    
    def _clear_visualization_paths(self):
        """Clear all visualized paths."""
        if self.meshcat is None:
            return
        self.meshcat.Delete("paths")
        self.path_line_count = 0
    
    # ==================== END VISUALIZATION METHODS ====================
    
    def add_regions_to_gcs(self, new_regions: List[HPolyhedron]) -> int:
        """
        Add new IRIS regions to the GCS planner's region list.
        
        Args:
            new_regions: List of HPolyhedron regions to add
            
        Returns:
            Number of regions actually added (after filtering redundant ones)
        """
        added_count = 0
        for region in new_regions:
            if not self._is_redundant_region(region):
                self.gcs_planner.regions.append(region)
                added_count += 1
        
        self.stats.total_regions_added += added_count
        self.stats.total_regions_after_pruning += added_count
        if self.logging:
            print(f"  [OnlineGCS] Added {added_count} new regions "
                  f"(total: {len(self.gcs_planner.regions)})", flush=True)
        return added_count
    
    def _is_redundant_region(self, new_region: HPolyhedron, 
                              containment_check: bool = True) -> bool:
        """
        Check if a new region is redundant (already covered by existing regions).
        
        Args:
            new_region: The region to check
            containment_check: If True, check if new region is contained in existing
            
        Returns:
            True if the region is redundant and should not be added
        """
        if not self.gcs_planner.regions:
            return False
            
        for existing in self.gcs_planner.regions:
            if containment_check:
                try:
                    center = new_region.ChebyshevCenter()
                    if existing.PointInSet(center):
                        samples_inside = 0
                        total_samples = 10
                        for _ in range(total_samples):
                            sample = new_region.UniformSample(self.rng)
                            if existing.PointInSet(sample):
                                samples_inside += 1
                        if samples_inside >= total_samples * 0.8:
                            return True
                except Exception:
                    pass
        return False

    def get_random_point_from_shelf_configs(self) -> np.ndarray:
        """Get a random target configuration from the shelf configs."""
        idx = self.rng.integers(0, len(self.shelf_configs))
        return idx

    def get_statistics(self) -> dict:
        """Return current statistics of the online GCS algorithm."""
        return {
            "total_regions": len(self.gcs_planner.regions),
            "gcs_success_count": self.gcs_success_count,
            "rrt_fallback_count": self.rrt_fallback_count,
            "total_queries": self.total_queries,
            "gcs_success_rate": self.gcs_success_count / max(1, self.total_queries),
            "total_regions_added": self.total_regions_added,
            "total_regions_after_pruning": self.total_regions_after_pruning,
        }
    
    def print_statistics(self):
        """Print current statistics."""
        stats = self.get_statistics()
        print(f"\n{'='*50}")
        print("Online GCS Statistics")
        print(f"{'='*50}")
        print(f"  Total queries:      {stats['total_queries']}")
        print(f"  GCS successes:      {stats['gcs_success_count']}")
        print(f"  RRT fallbacks:      {stats['rrt_fallback_count']}")
        print(f"  GCS success rate:   {stats['gcs_success_rate']*100:.1f}%")
        print(f"  Total regions:      {stats['total_regions']}")
        print(f"  Regions added:      {stats['total_regions_added']}")
        print(f"  Regions after pruning: {stats['total_regions_after_pruning']}")
        print(f"{'='*50}\n")
    
    def prune_redundant_regions(self) -> int:
        """
        Remove regions that are fully contained in other regions.
        
        Returns:
            Number of regions pruned
        """
        if len(self.gcs_planner.regions) <= 1:
            return 0
        
        non_redundant = []
        pruned_count = 0
        
        for i, region in enumerate(self.gcs_planner.regions):
            is_contained = False
            center = region.ChebyshevCenter()
            
            for j, other in enumerate(self.gcs_planner.regions):
                if i != j:
                    if other.PointInSet(center):
                        try:
                            samples_inside = 0
                            total_samples = 5
                            for _ in range(total_samples):
                                sample = region.UniformSample(self.rng)
                                if other.PointInSet(sample):
                                    samples_inside += 1
                            if samples_inside >= total_samples * 0.9:
                                is_contained = True
                                break
                        except Exception:
                            pass
            
            if not is_contained:
                non_redundant.append(region)
            else:
                pruned_count += 1
        
        self.gcs_planner.regions = non_redundant
        
        if self.logging and pruned_count > 0:
            print(f"  [OnlineGCS] Pruned {pruned_count} redundant regions", flush=True)
        
        return pruned_count

    def _load_cspace_positions(self, yaml_path: str) -> Dict[str, List[np.ndarray]]:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        result = {}
        for key, positions in data.items():
            result[key] = [np.array(p) for p in positions]
        
        return result

    def run(self, num_keypoints: int = 10, prune_interval: int = 20,
            animation_speed: float = 1.0):
        """
        Run the online GCS algorithm.
        
        Args:
            num_keypoints: Number of keypoints to extract from RRT path
            prune_interval: How often to prune redundant regions (0 = never)
            animation_speed: Speed multiplier for animations (higher = faster)
        """
        print(f"\n{'='*60}")
        print(f"Starting Online GCS for {self.scene_type.name}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Initial position: [{', '.join(f'{x:.2f}' for x in self.current_qpos[:3])}...]")
        if self.visualize:
            print(f"Visualization: ENABLED")
            print(f"Meshcat URL: {self.meshcat.web_url()}")
        print(f"{'='*60}\n")
        
        if self.visualize:
            self._update_robot_visualization(self.current_qpos)
            time.sleep(0.5)
        
        try:
            for iteration in range(self.max_iterations):
                self.total_queries += 1
                
                target_idx = self.get_random_point_from_shelf_configs()
                target_qpos = np.array(self.shelf_configs[target_idx])
                
                if self.logging:
                    print(f"\n[Iter {iteration + 1}/{self.max_iterations}] "
                          f"Target: [{', '.join(f'{x:.2f}' for x in target_qpos[:3])}...]")
                
                if self.visualize:
                    self._visualize_target_point(target_qpos)
                
                start_in_gcs = self.check_if_point_in_iris_regions(self.current_qpos)
                goal_in_gcs = self.check_if_point_in_iris_regions(target_qpos)
                
                if start_in_gcs and goal_in_gcs:
                    if self.logging:
                        print(f"  Both points in GCS, trying GCS planning...")
                    
                    success = self.gcs_planner.solve_from_configs(
                        self.current_qpos, target_qpos, build_missing_regions=False
                    )
                    
                    if success:
                        self.stats.gcs_success_count += 1
                        
                        if self.visualize and self.gcs_planner.trajectory is not None:
                            traj = self.gcs_planner.trajectory
                            times = np.linspace(traj.start_time(), traj.end_time(), 20)
                            path_samples = [traj.value(t).flatten() for t in times]
                            self._visualize_path(path_samples, use_gcs=True)
                            self._animate_trajectory(traj, duration=2.0 / animation_speed)
                        
                        self.current_qpos = target_qpos.copy()
                        if self.logging:
                            print(f"  GCS SUCCESS! Path length: {self.gcs_planner.path_length:.3f}")
                        continue
                    else:
                        if self.logging:
                            print(f"  GCS failed (regions not connected), falling back to RRT")
                
                self.stats.rrt_fallback_count += 1
                
                if self.logging:
                    print(f"  Running RRT* from current to target...")
                
                rrt_path = self.rrt_planner.plan_bidirectional(
                    self.current_qpos, target_qpos, goal_tolerance=0.15
                )
                if rrt_path is None:
                    rrt_path = self.rrt_planner.plan(
                        self.current_qpos, target_qpos, goal_tolerance=0.15
                    )
                
                if rrt_path is not None:
                    if self.logging:
                        print(f"  RRT found path with {len(rrt_path)} points, "
                              f"length: {self.rrt_planner.path_length:.3f}")
                    
                    if self.visualize:
                        self._visualize_path(rrt_path, use_gcs=False)
                    
                    keypoints = extract_keypoints_uniform(rrt_path, num_points=num_keypoints)
                    
                    if self.logging:
                        print(f"  Building IRIS regions from {len(keypoints)} keypoints...")
                    
                    build_time = self.iris_region_builder.build_regions_from_seeds(keypoints)
                    
                    new_regions = self.iris_region_builder.regions
                    added = self.add_regions_to_gcs(new_regions)
                    
                    if self.logging:
                        print(f"  Added {added} regions in {build_time:.2f}s")
                    
                    success = self.gcs_planner.solve_from_configs(
                        self.current_qpos, target_qpos, build_missing_regions=True
                    )
                    
                    if success:
                        if self.visualize and self.gcs_planner.trajectory is not None:
                            self._animate_trajectory(
                                self.gcs_planner.trajectory, 
                                duration=2.0 / animation_speed
                            )
                        
                        self.current_qpos = target_qpos.copy()
                        if self.logging:
                            print(f"  GCS with new regions SUCCESS! "
                                  f"Path length: {self.gcs_planner.path_length:.3f}")
                    else:
                        if self.visualize:
                            self._animate_path(rrt_path, duration=2.0 / animation_speed)
                        
                        self.current_qpos = target_qpos.copy()
                        if self.logging:
                            print(f"  GCS still failed, using RRT path to reach target")
                else:
                    print(f"  [WARNING] RRT failed to find path from current to target")
                    continue
                
                if self.visualize:
                    self._update_robot_visualization(self.current_qpos)
                
                if prune_interval > 0 and (iteration + 1) % prune_interval == 0:
                    pruned = self.prune_redundant_regions()
                    if self.logging and pruned > 0:
                        print(f"  Pruned {pruned} redundant regions")
                
                if (iteration + 1) % 10 == 0:
                    print(f"[Progress] Iteration {iteration + 1}/{self.max_iterations}, "
                          f"Regions: {len(self.gcs_planner.regions)}, "
                          f"GCS rate: {self.gcs_success_count}/{self.total_queries} "
                          f"({self.gcs_success_count/self.total_queries*100:.1f}%)")
                    
        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Stopping online GCS...")
        
        self.print_statistics()
        return self.get_statistics()
    
    def save_regions(self, filepath: str):
        """Save current IRIS regions to a YAML file."""
        from pydrake.all import SaveIrisRegionsYamlFile
        regions_dict = {f"region_{i}": r for i, r in enumerate(self.gcs_planner.regions)}
        SaveIrisRegionsYamlFile(filepath, regions_dict)
        print(f"Saved {len(self.gcs_planner.regions)} regions to {filepath}")


def main():
    import logging
    logging.getLogger("drake").setLevel(logging.WARNING)
    
    parser = argparse.ArgumentParser(description="Online GCS Region Building")
    parser.add_argument("--scene", type=str, default="SINGLE_SHELF",
                        choices=["SINGLE_SHELF", "TWO_SHELVES", "TABLE_THREE_SHELVES"],
                        help="Scene type (default: SINGLE_SHELF)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Max iterations (default: 50)")
    parser.add_argument("--keypoints", type=int, default=10,
                        help="Number of keypoints per RRT path (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--prune-interval", type=int, default=20,
                        help="Prune redundant regions every N iterations (default: 20, 0=never)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for saving regions (optional)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable Meshcat visualization")
    parser.add_argument("--animation-speed", type=float, default=1.0,
                        help="Animation speed multiplier (default: 1.0, higher=faster)")
    args = parser.parse_args()
    
    scene_type = SceneType[args.scene]
    
    print("=" * 60)
    print("Online GCS Region Building")
    print("=" * 60)
    print(f"Scene:          {scene_type.name}")
    print(f"Iterations:     {args.iterations}")
    print(f"Keypoints:      {args.keypoints}")
    print(f"Random seed:    {args.seed}")
    print(f"Prune interval: {args.prune_interval}")
    print(f"Verbose:        {args.verbose}")
    print(f"Visualize:      {args.visualize}")
    print("=" * 60)
    
    online_gcs = OnlineGCS(
        scene_type=scene_type,
        random_seed=args.seed,
        logging=args.verbose,
        visualize=args.visualize,
        max_iterations=args.iterations,
    )
    
    stats = online_gcs.run(
        num_keypoints=args.keypoints,
        prune_interval=args.prune_interval,
        animation_speed=args.animation_speed,
    )
    
    if args.output:
        online_gcs.save_regions(args.output)
    
    return stats


if __name__ == "__main__":
    main()