from dataclasses import dataclass, field
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
from helpers.parallel_exploration import ParallelExplorationCoordinator
from helpers.rrt_star_planner import RRTStarPlanner
from helpers.rrt_star_planner import extract_keypoints_uniform
from helpers.opt_solver import TrajOptSolver

@dataclass
class OnlineGCSStats:
    total_regions: int = 0
    gcs_success_count: int = 0
    rrt_fallback_count: int = 0
    total_queries: int = 0
    gcs_success_rate: float = 0.0
    total_regions_added: int = 0
    total_regions_after_pruning: int = 0
    total_keypoints_requested: int = 0
    total_iris_regions_built: int = 0
    warmstart_time: float = 0.0
    warmstart_regions: int = 0
    stats_per_way_iris_build_time: Dict[str, List[float]] = field(default_factory=dict)
    # Dict[way_name, [path_lengths, times]]
    stats_per_each_query_gcs: Dict[str, List[List[float]]] = field(default_factory=dict)
    stats_per_each_query_rrt: Dict[str, List[List[float]]] = field(default_factory=dict)
    # Dict[way_name, [path_lengths, times, successes]]
    stats_per_each_query_trajopt: Dict[str, List[List[float]]] = field(default_factory=dict)
    # Dict[way_name, List[dict]] — quality metrics per query (jerk, smoothness, energy, max_accel)
    stats_gcs_quality: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    stats_trajopt_quality: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)


class OnlineGCS:
    def __init__(self, scene_type: SceneType, random_seed: int = 42,
                 logging: bool = False, visualize: bool = False, smart_keypoints: bool = False,
                 max_iterations: int = 100, parallel_exploration: bool = False, num_workers: int = 2,
                 k_shortest_paths: int = 1,
                 warmstart: bool = False, warmstart_seeds: int = 15):
        self.scene_type = scene_type
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.logging = logging
        self.visualize = visualize
        self.max_iterations = max_iterations
        self.smart_keypoints = smart_keypoints
        self.parallel_exploration = parallel_exploration
        self.num_workers = num_workers
        self.k_shortest_paths = k_shortest_paths
        self.warmstart = warmstart
        self.warmstart_seeds = warmstart_seeds

        self.exploration_coordinator: Optional[ParallelExplorationCoordinator] = None
        if parallel_exploration:
            self.exploration_coordinator = ParallelExplorationCoordinator(
                scene_type=scene_type,
                num_workers=num_workers,
                random_seed=random_seed,
            )

        self.iris_region_builder = IRISRegionBuilder(scene_type=scene_type, random_seed=random_seed)
        
        self.gcs_planner = GCSPathPlanner.from_iris_builder(
            self.iris_region_builder, k_shortest_paths=k_shortest_paths
        )
        
        self.rrt_planner = RRTStarPlanner(
            scene_type=scene_type,
            random_seed=random_seed,
        )
        
        self.trajopt_solver = TrajOptSolver(
            scene_type=scene_type,
            num_knots=21,
            d_min=0.01,
            smoothness_weight=1.0,
            path_length_weight=0.1,
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
        self.current_idx = -1

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

    def _compute_trajectory_metrics(self, traj, num_samples: int = 500) -> Dict[str, float]:
        """Compute *arc-length-parameterised* quality metrics so that the
        comparison does not depend on how fast the trajectory is traversed.

        Returns curvature, torsion (geometric).
        """
        t0 = traj.start_time()
        tf = traj.end_time()
        duration = tf - t0
        zeros = {"curv_max": 0.0, "curv_integral": 0.0,
                 "torsion_max": 0.0, "torsion_integral": 0.0,
                 "path_length": 0.0}
        if duration < 1e-10:
            return zeros

        times = np.linspace(t0, tf, num_samples)
        positions = np.column_stack([traj.value(t).flatten() for t in times])

        seg_lens = np.linalg.norm(np.diff(positions, axis=1), axis=0)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]
        if total_len < 1e-10:
            return zeros

        target_s = np.linspace(0.0, total_len, num_samples)
        nq = positions.shape[0]
        resampled = np.zeros((nq, num_samples))
        for i in range(num_samples):
            idx = int(np.searchsorted(cum_len, target_s[i], side="right")) - 1
            idx = min(max(idx, 0), len(cum_len) - 2)
            seg = cum_len[idx + 1] - cum_len[idx]
            alpha = (target_s[i] - cum_len[idx]) / seg if seg > 1e-12 else 0.0
            resampled[:, i] = (1.0 - alpha) * positions[:, idx] + alpha * positions[:, idx + 1]

        ds = target_s[1] - target_s[0] if num_samples > 1 else 1.0
        d1 = np.diff(resampled, axis=1) / ds
        d2 = np.diff(d1, axis=1) / ds
        d3 = np.diff(d2, axis=1) / ds

        curv = np.linalg.norm(d2, axis=0)
        tors = np.linalg.norm(d3, axis=0)

        return {
            "curv_max": float(np.max(curv)) if len(curv) else 0.0,
            "curv_integral": float(np.trapz(curv, dx=ds)),
            "torsion_max": float(np.max(tors)) if len(tors) else 0.0,
            "torsion_integral": float(np.trapz(tors, dx=ds)),
            "path_length": float(total_len),
        }

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
            "warmstart_time": self.stats.warmstart_time,
            "warmstart_regions": self.stats.warmstart_regions,
            "regions_in_gcs": len(self.gcs_planner.regions),
            "total_regions": self.stats.total_keypoints_requested,
            "gcs_success_count": self.stats.gcs_success_count,
            "rrt_fallback_count": self.stats.rrt_fallback_count,
            "total_queries": self.stats.total_queries,
            "gcs_success_rate": self.stats.gcs_success_count / max(1, self.stats.total_queries),
            "total_regions_added": self.stats.total_iris_regions_built,
            "total_regions_after_pruning": self.stats.total_regions_after_pruning,
            "ways_rrt_time": {way: np.mean(self.stats.stats_per_each_query_rrt[way][1]) for way in self.stats.stats_per_each_query_rrt},
            "ways_gcs_time": {way: np.mean(self.stats.stats_per_each_query_gcs[way][1]) for way in self.stats.stats_per_each_query_gcs},
            "ways_gcs_vanilla_time": {way: np.mean(self.stats.stats_per_each_query_gcs[way][2]) for way in self.stats.stats_per_each_query_gcs},
            "ways_rrt_path_length": {way: np.mean(self.stats.stats_per_each_query_rrt[way][0]) for way in self.stats.stats_per_each_query_rrt},
            "ways_gcs_path_length": {way: np.mean(self.stats.stats_per_each_query_gcs[way][0]) for way in self.stats.stats_per_each_query_gcs},
            "ways_iris_build_time": {way: np.mean(self.stats.stats_per_way_iris_build_time[way]) for way in self.stats.stats_per_way_iris_build_time},
            "ways_trajopt_time": {way: np.mean(self.stats.stats_per_each_query_trajopt[way][1]) for way in self.stats.stats_per_each_query_trajopt},
            "ways_trajopt_path_length": {
                way: np.mean([l for l, s in zip(self.stats.stats_per_each_query_trajopt[way][0],
                                                 self.stats.stats_per_each_query_trajopt[way][2]) if s > 0.5])
                if any(s > 0.5 for s in self.stats.stats_per_each_query_trajopt[way][2]) else 0.0
                for way in self.stats.stats_per_each_query_trajopt
            },
            "ways_trajopt_success_rate": {
                way: np.mean(self.stats.stats_per_each_query_trajopt[way][2])
                for way in self.stats.stats_per_each_query_trajopt
            },
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
        print(f"  Total regions:      {stats['total_regions']} (keypoints requested for building)")
        print(f"  Regions added:      {stats['total_regions_added']} (IRIS regions actually built)")
        print(f"  Regions in GCS:     {stats['regions_in_gcs']} (current graph)")
        if stats['warmstart_regions'] > 0:
            print(f"  Warmstart regions:  {stats['warmstart_regions']} (in {stats['warmstart_time']:.1f}s)")

        all_ways = (
            set(self.stats.stats_per_each_query_rrt.keys())
            | set(self.stats.stats_per_each_query_gcs.keys())
            | set(self.stats.stats_per_way_iris_build_time.keys())
            | set(self.stats.stats_per_each_query_trajopt.keys())
        )
        for way in sorted(all_ways):
            print(f"\n  --- {way} ---")
            rrt_time = stats["ways_rrt_time"].get(way)
            gcs_time = stats["ways_gcs_time"].get(way)
            gcs_vanilla_time = stats["ways_gcs_vanilla_time"].get(way)
            rrt_path_len = stats["ways_rrt_path_length"].get(way)
            gcs_path_len = stats["ways_gcs_path_length"].get(way)
            iris_build_time = stats["ways_iris_build_time"].get(way)
            trajopt_time = stats["ways_trajopt_time"].get(way)
            trajopt_path_len = stats["ways_trajopt_path_length"].get(way)
            trajopt_success = stats["ways_trajopt_success_rate"].get(way)

            if rrt_time is not None:
                print(f"  RRT time for {way}: {rrt_time:.5f}s")
            else:
                print(f"  RRT time for {way}: N/A")

            if gcs_time is not None:
                print(f"  GCS time for {way}: {gcs_time:.5f}s")
            else:
                print(f"  GCS time for {way}: N/A")

            if gcs_vanilla_time is not None:
                print(f"  GCS vanilla time for {way}: {gcs_vanilla_time:.5f}s")
            else:
                print(f"  GCS vanilla time for {way}: N/A")

            if trajopt_time is not None:
                print(f"  Opt time for {way}: {trajopt_time:.5f}s")
            else:
                print(f"  Opt time for {way}: N/A")

            if rrt_path_len is not None:
                print(f"  RRT path length for {way}: {rrt_path_len:.2f}")
            else:
                print(f"  RRT path length for {way}: N/A")

            if gcs_path_len is not None:
                print(f"  GCS path length for {way}: {gcs_path_len:.2f}")
            else:
                print(f"  GCS path length for {way}: N/A")

            if trajopt_path_len is not None and trajopt_path_len > 0:
                print(f"  Optimization path length for {way}: {trajopt_path_len:.2f}")
            else:
                print(f"  Optimization path length for {way}: N/A")

            if trajopt_success is not None:
                print(f"  Optimization success rate for {way}: {trajopt_success*100:.0f}%")

            if iris_build_time is not None:
                print(f"  IRIS build time for {way}: {iris_build_time:.5f}s")
            else:
                print(f"  IRIS build time for {way}: 0.00s (regions from other paths)")

            # --- Trajectory quality metrics (arc-length-normalised) ---
            gcs_qlist = self.stats.stats_gcs_quality.get(way, [])
            topt_qlist = self.stats.stats_trajopt_quality.get(way, [])
            if gcs_qlist or topt_qlist:
                for metric_key, label in [
                    ("curv_max",         "Max curvature"),
                    ("curv_integral",    "Total curvature (∫κ ds)"),
                    ("torsion_max",      "Max torsion"),
                    ("torsion_integral", "Total torsion"),
                ]:
                    gcs_val = np.mean([m[metric_key] for m in gcs_qlist]) if gcs_qlist else None
                    topt_val = np.mean([m[metric_key] for m in topt_qlist]) if topt_qlist else None
                    gcs_s = f"{gcs_val:.4f}" if gcs_val is not None else "N/A"
                    topt_s = f"{topt_val:.4f}" if topt_val is not None else "N/A"
                    print(f"  {label}: GCS={gcs_s}  Opt={topt_s}")

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

    # ==================== WARMSTART ====================

    def _farthest_point_seeds(self, candidates: List[np.ndarray],
                              n_select: int,
                              initial_seeds: List[np.ndarray]) -> List[np.ndarray]:
        """Pick *n_select* points from *candidates* that are maximally
        spread out, given *initial_seeds* as already-placed centres.
        Returns initial_seeds + newly chosen points.
        """
        selected = [s.copy() for s in initial_seeds]
        if n_select <= 0 or not candidates:
            return selected

        cand = np.array(candidates)
        used = np.zeros(len(cand), dtype=bool)

        for _ in range(n_select):
            if np.all(used):
                break
            sel_arr = np.array(selected)
            dists = np.min(
                np.linalg.norm(
                    cand[:, np.newaxis, :] - sel_arr[np.newaxis, :, :], axis=2
                ),
                axis=1,
            )
            dists[used] = -1.0
            best = int(np.argmax(dists))
            if dists[best] <= 0:
                break
            selected.append(cand[best].copy())
            used[best] = True

        return selected

    def _run_warmstart(self):
        """Build initial IRIS regions using farthest-point sampling."""
        print(f"\n{'='*60}")
        print(f"Warmstart: building up to {self.warmstart_seeds} initial IRIS regions")
        print(f"{'='*60}")

        t0 = time.time()

        initial_seeds: List[np.ndarray] = []
        for cfg in self.shelf_configs:
            q = np.array(cfg)
            if self.iris_region_builder.is_collision_free(q):
                initial_seeds.append(q)

        if self.logging:
            print(f"  {len(initial_seeds)} shelf configs as initial seeds", flush=True)

        n_candidates = max(500, self.warmstart_seeds * 50)
        candidates = self.iris_region_builder.sample_collision_free(
            n_candidates, self.rng
        )
        if self.logging:
            print(f"  Sampled {len(candidates)} collision-free candidates", flush=True)

        n_extra = max(0, self.warmstart_seeds - len(initial_seeds))
        all_seeds = self._farthest_point_seeds(candidates, n_extra, initial_seeds)

        if self.logging:
            print(
                f"  Selected {len(all_seeds)} seeds "
                f"({len(initial_seeds)} shelf + {len(all_seeds) - len(initial_seeds)} farthest-point)",
                flush=True,
            )

        build_time = self.iris_region_builder.build_regions_from_seeds(all_seeds)
        added = self.add_regions_to_gcs(self.iris_region_builder.regions)

        total_time = time.time() - t0
        self.stats.warmstart_time = total_time
        self.stats.warmstart_regions = added

        print(
            f"Warmstart done: {added} regions in {total_time:.1f}s "
            f"(IRIS build: {build_time:.1f}s)",
            flush=True,
        )
        print(f"{'='*60}\n")
        return added, total_time

    # ==================== MAIN LOOP ====================

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


        if self.warmstart:
            self._run_warmstart()

        if self.visualize:
            self._update_robot_visualization(self.current_qpos)
            time.sleep(0.5)
        
        try:
            overall_time = time.time()
            
            for iteration in range(self.max_iterations):
                # Harvest any regions built by parallel workers
                if self.exploration_coordinator is not None:
                    parallel_regions = self.exploration_coordinator.collect_results(
                        self.gcs_planner.regions, self.rng
                    )
                    if parallel_regions:
                        added = self.add_regions_to_gcs(parallel_regions)
                        if self.logging and added > 0:
                            print(f"  [Parallel] Integrated {added} regions from workers", flush=True)

                self.stats.total_queries += 1
                
                target_idx = self.get_random_point_from_shelf_configs()
                target_qpos = np.array(self.shelf_configs[target_idx])
                
                if self.logging:
                    print(f"\n[Iter {iteration + 1}/{self.max_iterations}] "
                          f"Target: [{', '.join(f'{x:.2f}' for x in target_qpos[:3])}...]")
                
                if self.visualize:
                    self._visualize_target_point(target_qpos)

                current_way_name = f'{self.current_idx}-{target_idx}'
                # ================== start opt baseline ================
                rrt_start_time = time.time()
                rrt_path = self.rrt_planner.plan_bidirectional(
                    self.current_qpos, target_qpos, goal_tolerance=0.15
                )
                if rrt_path is None:
                    rrt_path = self.rrt_planner.plan(
                        self.current_qpos, target_qpos, goal_tolerance=0.15
                    )
                rrt_time = time.time() - rrt_start_time
                if current_way_name not in self.stats.stats_per_each_query_rrt:
                    self.stats.stats_per_each_query_rrt[current_way_name] = [[], []]
                self.stats.stats_per_each_query_rrt[current_way_name][0].append(
                    self.rrt_planner.path_length if rrt_path is not None else 0.0
                )
                self.stats.stats_per_each_query_rrt[current_way_name][1].append(rrt_time)

                # --- Optimization baseline: optimize the RRT path ---
                if rrt_path is not None and len(rrt_path) >= 2:
                    trajopt_result = self.trajopt_solver.optimize_rrt_path(rrt_path)
                    if current_way_name not in self.stats.stats_per_each_query_trajopt:
                        self.stats.stats_per_each_query_trajopt[current_way_name] = [[], [], []]
                    self.stats.stats_per_each_query_trajopt[current_way_name][0].append(
                        trajopt_result.path_length if trajopt_result.success else 0.0
                    )
                    self.stats.stats_per_each_query_trajopt[current_way_name][1].append(
                        trajopt_result.solve_time
                    )
                    self.stats.stats_per_each_query_trajopt[current_way_name][2].append(
                        float(trajopt_result.success)
                    )
                    if trajopt_result.success and trajopt_result.trajectory is not None:
                        topt_qm = self._compute_trajectory_metrics(trajopt_result.trajectory)
                        self.stats.stats_trajopt_quality.setdefault(current_way_name, []).append(topt_qm)
                    if self.visualize and trajopt_result.success and trajopt_result.path:
                        trajopt_color = Rgba(0.8, 0.2, 0.8, 0.7)
                        self._visualize_path(trajopt_result.path, color=trajopt_color)
                    if self.logging:
                        if trajopt_result.success:
                            print(f"  TrajOpt path length: {trajopt_result.path_length:.3f} "
                                  f"(was {trajopt_result.initial_path_length:.3f}), "
                                  f"time: {trajopt_result.solve_time:.3f}s")
                        else:
                            print(f"  TrajOpt FAILED, time: {trajopt_result.solve_time:.3f}s")

                start_in_gcs = self.check_if_point_in_iris_regions(self.current_qpos)
                # ================== end opt baseline ================
                goal_in_gcs = self.check_if_point_in_iris_regions(target_qpos)

                if goal_in_gcs:
                    if self.logging:
                        print(f"  Both points in GCS, trying GCS planning...")
                    
                    gcs_start_time = time.time()
                    success = self.gcs_planner.solve_from_configs(
                        self.current_qpos, target_qpos, build_missing_regions=True ## !! changed to true for testing
                    )
                    gcs_time = time.time() - gcs_start_time
                    # Vanilla GCS solve time is stored on the planner.
                    gcs_vanilla_time = self.gcs_planner.solve_time
                    
                    if success:
                        self.stats.gcs_success_count += 1
                        
                        if self.visualize and self.gcs_planner.trajectory is not None:
                            traj = self.gcs_planner.trajectory
                            times = np.linspace(traj.start_time(), traj.end_time(), 20)
                            path_samples = [traj.value(t).flatten() for t in times]
                            self._visualize_path(path_samples, use_gcs=True)
                            self._animate_trajectory(traj, duration=2.0 / animation_speed)
                        
                        current_way_name = f'{self.current_idx}-{target_idx}'
                        if current_way_name not in self.stats.stats_per_each_query_gcs:
                            self.stats.stats_per_each_query_gcs[current_way_name] = [[], [], []]
                        self.stats.stats_per_each_query_gcs[current_way_name][0].append(self.gcs_planner.path_length)
                        self.stats.stats_per_each_query_gcs[current_way_name][1].append(gcs_time)
                        self.stats.stats_per_each_query_gcs[current_way_name][2].append(gcs_vanilla_time)
                        if current_way_name not in self.stats.stats_per_way_iris_build_time:
                            self.stats.stats_per_way_iris_build_time[current_way_name] = []
                        self.stats.stats_per_way_iris_build_time[current_way_name].append(0.0)

                        if self.gcs_planner.trajectory is not None:
                            gcs_qm = self._compute_trajectory_metrics(self.gcs_planner.trajectory)
                            self.stats.stats_gcs_quality.setdefault(current_way_name, []).append(gcs_qm)

                        self.current_qpos = target_qpos.copy()
                        self.current_idx = target_idx
                        if self.logging:
                            print(f"  GCS SUCCESS! Path length: {self.gcs_planner.path_length:.3f}")
                        continue
                    else:
                        if self.logging:
                            print(f"  GCS failed (regions not connected), falling back to RRT")

                self.stats.rrt_fallback_count += 1

                if self.logging:
                    print(f"  Using RRT* path from current to target...")

                rrt_path = self.rrt_planner.plan_bidirectional(
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
                    
                    self.stats.total_keypoints_requested += len(keypoints)
                    build_time = self.iris_region_builder.build_regions_from_seeds(
                        keypoints, existing_regions=self.gcs_planner.regions
                    )
                    num_built = len(self.iris_region_builder.regions)
                    self.stats.total_iris_regions_built += num_built
                    if current_way_name not in self.stats.stats_per_way_iris_build_time:
                        self.stats.stats_per_way_iris_build_time[current_way_name] = []
                    self.stats.stats_per_way_iris_build_time[current_way_name].append(build_time)
                    new_regions = self.iris_region_builder.regions
                    added = self.add_regions_to_gcs(new_regions)
                    
                    if self.logging:
                        print(f"  Added {added} regions in {build_time:.2f}s")
                    
                    # Submit exploration tasks for workers to build regions in uncovered space
                    if self.exploration_coordinator is not None:
                        q_lower = self.iris_region_builder.q_lower
                        q_upper = self.iris_region_builder.q_upper
                        seeds = self.exploration_coordinator.generate_exploration_seeds(
                            self.gcs_planner.regions, q_lower, q_upper, self.rng, count=min(4, self.num_workers * 2)
                        )
                        if seeds:
                            self.exploration_coordinator.submit_tasks(seeds, interval_id=iteration)
                            if self.logging:
                                print(f"  [Parallel] Submitted {len(seeds)} exploration tasks", flush=True)
                    
                    gcs_start_time = time.time()
                    success = self.gcs_planner.solve_from_configs(
                        self.current_qpos, target_qpos, build_missing_regions=True
                    )
                    gcs_time = time.time() - gcs_start_time
                    gcs_vanilla_time = self.gcs_planner.solve_time

                    if success:
                        if self.visualize and self.gcs_planner.trajectory is not None:
                            traj = self.gcs_planner.trajectory
                            times = np.linspace(traj.start_time(), traj.end_time(), 20)
                            path_samples = [traj.value(t).flatten() for t in times]
                            self._visualize_path(path_samples, use_gcs=True)
                            self._animate_trajectory(
                                self.gcs_planner.trajectory, 
                                duration=2.0 / animation_speed
                            )

                        if current_way_name not in self.stats.stats_per_each_query_gcs:
                            self.stats.stats_per_each_query_gcs[current_way_name] = [[], [], []]
                        self.stats.stats_per_each_query_gcs[current_way_name][0].append(self.gcs_planner.path_length)
                        self.stats.stats_per_each_query_gcs[current_way_name][1].append(gcs_time)
                        self.stats.stats_per_each_query_gcs[current_way_name][2].append(gcs_vanilla_time)

                        if self.gcs_planner.trajectory is not None:
                            gcs_qm = self._compute_trajectory_metrics(self.gcs_planner.trajectory)
                            self.stats.stats_gcs_quality.setdefault(current_way_name, []).append(gcs_qm)

                        self.current_qpos = target_qpos.copy()
                        self.current_idx = target_idx
                        if self.logging:
                            print(f"  GCS with new regions SUCCESS! "
                                  f"Path length: {self.gcs_planner.path_length:.3f}")
                    else:
                        if self.visualize:
                            self._animate_path(rrt_path, duration=2.0 / animation_speed)
                        # RRT already recorded at start of iteration

                        self.current_qpos = target_qpos.copy()
                        self.current_idx = target_idx
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
                          f"GCS rate: {self.stats.gcs_success_count}/{self.stats.total_queries} "
                          f"({self.stats.gcs_success_count/self.stats.total_queries*100:.1f}%)")
            overall_time = time.time() - overall_time
            print(f"########### Overall time: {overall_time:.2f}s")      
        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Stopping online GCS...")
        finally:
            if self.exploration_coordinator is not None:
                self.exploration_coordinator.shutdown()
                if self.logging:
                    print("  [Parallel] Workers shut down", flush=True)
        
        self.print_statistics()
        return self.get_statistics()
    
    def save_regions(self, filepath: str):
        """Save current IRIS regions to a YAML file."""
        from pydrake.all import SaveIrisRegionsYamlFile
        regions_dict = {f"region_{i}": r for i, r in enumerate(self.gcs_planner.regions)}
        SaveIrisRegionsYamlFile(filepath, regions_dict)
        print(f"Saved {len(self.gcs_planner.regions)} regions to {filepath}")

    def run_rrt_only(self):
        if self.visualize:
            print(f"Visualization: ENABLED")
            print(f"Meshcat URL: {self.meshcat.web_url()}")
        print(f"{'='*60}\n")
        
        if self.visualize:
            self._update_robot_visualization(self.current_qpos)
            time.sleep(0.5)
        
        try:
            overall_time = time.time()
            for iteration in range(self.max_iterations):
                

                self.stats.total_queries += 1
                
                target_idx = self.get_random_point_from_shelf_configs()
                target_qpos = np.array(self.shelf_configs[target_idx])
                
                if self.logging:
                    print(f"\n[Iter {iteration + 1}/{self.max_iterations}] "
                          f"Target: [{', '.join(f'{x:.2f}' for x in target_qpos[:3])}...]")
                
                if self.visualize:
                    self._visualize_target_point(target_qpos)

                rrt_path = self.rrt_planner.plan_bidirectional(
                    self.current_qpos, target_qpos, goal_tolerance=0.15
                )

                if rrt_path is not None:
                    print(iteration)

                if self.visualize:
                        self._visualize_path(rrt_path, use_gcs=False)

            overall_time = time.time() - overall_time
            print(f"########### Overall time: {overall_time:.2f}s")  
        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Stopping online GCS...")

    def run_opt_only(self):
        if self.visualize:
            print(f"Visualization: ENABLED")
            print(f"Meshcat URL: {self.meshcat.web_url()}")
        print(f"{'='*60}\n")
        
        if self.visualize:
            self._update_robot_visualization(self.current_qpos)
            time.sleep(0.5)
        
        try:
            overall_time = time.time()
            for iteration in range(self.max_iterations):

                self.stats.total_queries += 1
                
                target_idx = self.get_random_point_from_shelf_configs()
                target_qpos = np.array(self.shelf_configs[target_idx])
                
                if self.logging:
                    print(f"\n[Iter {iteration + 1}/{self.max_iterations}] "
                          f"Target: [{', '.join(f'{x:.2f}' for x in target_qpos[:3])}...]")
                
                if self.visualize:
                    self._visualize_target_point(target_qpos)

                rrt_path = self.rrt_planner.plan_bidirectional(
                    self.current_qpos, target_qpos, goal_tolerance=0.15
                )

                if rrt_path is not None and len(rrt_path) >= 2:
                    trajopt_result = self.trajopt_solver.optimize_rrt_path(rrt_path)
                    if self.visualize and trajopt_result.success and trajopt_result.path:
                        trajopt_color = Rgba(0.8, 0.2, 0.8, 0.7)
                        self._visualize_path(trajopt_result.path, color=trajopt_color)
                    if self.logging:
                        if trajopt_result.success:
                            print(f"  TrajOpt path length: {trajopt_result.path_length:.3f} "
                                  f"(was {trajopt_result.initial_path_length:.3f}), "
                                  f"time: {trajopt_result.solve_time:.3f}s")
                        else:
                            print(f"  TrajOpt FAILED, time: {trajopt_result.solve_time:.3f}s")

                if self.visualize:
                        self._visualize_path(rrt_path, use_gcs=False)

            overall_time = time.time() - overall_time
            print(f"########### Overall time: {overall_time:.2f}s")  
        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Stopping online GCS...")


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
    parser.add_argument("--smart-keypoints", action="store_true",
                        help="Use smart keypoints") # TODO
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel IRIS region exploration (background workers)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of parallel exploration workers when --parallel (default: 2)")
    parser.add_argument("--k-shortest-paths", type=int, default=3,
                        help="Number of shortest paths (Yen's algorithm) for GCS subgraph selection (default: 3)")
    parser.add_argument("--warmstart", action="store_true",
                        help="Build initial IRIS regions before main loop (farthest-point sampling)")
    parser.add_argument("--warmstart-seeds", type=int, default=15,
                        help="Number of seed points for warmstart IRIS regions (default: 15)")
    parser.add_argument("--rrt-only", action="store_true",
                        help="Run only RRT* path planning")
    parser.add_argument("--opt-only", action="store_true",
                        help="Run only optimization of RRT* path")
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
    print(f"Parallel:       {args.parallel}" + (f" ({args.num_workers} workers)" if args.parallel else ""))
    print(f"K shortest:     {args.k_shortest_paths}")
    print(f"Warmstart:      {args.warmstart}" + (f" ({args.warmstart_seeds} seeds)" if args.warmstart else ""))
    print(f"RRT only:      {args.rrt_only}")
    print(f"Opt only:      {args.opt_only}")
    print("=" * 60)
    
    online_gcs = OnlineGCS(
        scene_type=scene_type,
        random_seed=args.seed,
        logging=args.verbose,
        visualize=args.visualize,
        max_iterations=args.iterations,
        parallel_exploration=args.parallel,
        num_workers=args.num_workers,
        k_shortest_paths=args.k_shortest_paths,
        warmstart=args.warmstart,
        warmstart_seeds=args.warmstart_seeds,
    )
    
    if args.rrt_only:
        online_gcs.run_rrt_only()
    elif args.opt_only:
        online_gcs.run_opt_only()
    else:
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