from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from pydrake.all import HPolyhedron, Rgba

from online_gcs.exploration.parallel import ParallelExplorationCoordinator
from online_gcs.metrics import compute_trajectory_metrics
from online_gcs.planners.gcs import GCSPathPlanner
from online_gcs.planners.rrt_star import RRTStarPlanner, extract_keypoints_uniform
from online_gcs.planners.trajopt import TrajOptSolver
from online_gcs.regions.iris import IRISRegionBuilder
from online_gcs.scenes import SceneType, load_shelf_configurations
from online_gcs.visualization import MeshcatVisualizer


def choose_target_index(
    count: int,
    current_idx: int,
    rng: np.random.Generator,
) -> int:
    """Choose a target configuration other than the current one."""
    if count < 2:
        raise ValueError("at least two target configurations are required")
    candidates = np.delete(np.arange(count), current_idx)
    return int(rng.choice(candidates))


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
    def __init__(
        self,
        scene_type: SceneType,
        random_seed: int = 42,
        logging: bool = False,
        visualize: bool = False,
        smart_keypoints: bool = False,
        max_iterations: int = 100,
        parallel_exploration: bool = False,
        num_workers: int = 2,
        k_shortest_paths: int = 1,
        warmstart: bool = False,
        warmstart_seeds: int = 15,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

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

        self.shelf_configs = load_shelf_configurations(scene_type)

        self.current_qpos = np.array(self.shelf_configs[0]) if self.shelf_configs else np.zeros(7)
        self.current_idx = 0

        self.stats = OnlineGCSStats()

        self.visualizer = MeshcatVisualizer(scene_type, self.current_qpos) if visualize else None

    def check_if_point_in_iris_regions(self, point: np.ndarray) -> bool:
        """Check if a configuration point is inside any existing IRIS region."""
        if point is None:
            return False
        return self.gcs_planner._find_containing_region(point) != -1

    def _pick_paper_figure_target(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Pick a target that triggers online expansion and yields a clear demo."""
        best_idx = None
        best_target = None
        best_score = -np.inf

        for idx, cfg in enumerate(self.shelf_configs):
            target_qpos = np.array(cfg)
            if np.allclose(target_qpos, self.current_qpos):
                continue

            goal_in_gcs = self.check_if_point_in_iris_regions(target_qpos)
            gcs_success = False
            if goal_in_gcs:
                try:
                    gcs_success = self.gcs_planner.solve_from_configs(
                        self.current_qpos,
                        target_qpos,
                        build_missing_regions=False,
                    )
                except Exception:
                    gcs_success = False

            if goal_in_gcs and gcs_success:
                continue

            rrt_path = self.rrt_planner.plan_bidirectional(
                self.current_qpos, target_qpos, goal_tolerance=0.15
            )
            if rrt_path is None:
                rrt_path = self.rrt_planner.plan(
                    self.current_qpos, target_qpos, goal_tolerance=0.15
                )
            if rrt_path is None:
                continue

            score = float(self.rrt_planner.path_length)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_target = target_qpos

        return best_idx, best_target

    def _hold_figure_scene(self, hold_seconds: float):
        """Keep Meshcat alive long enough to inspect or screenshot the figure."""
        if self.visualizer is None:
            return
        if hold_seconds > 0:
            print(f"Holding figure scene for {hold_seconds:.1f}s...", flush=True)
            time.sleep(hold_seconds)
            return

        print(
            "Figure scene is ready. Keep this process running while you inspect Meshcat.",
            flush=True,
        )
        print("Press Ctrl+C in this terminal when you are done.", flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Closing paper figure demo...", flush=True)

    def run_paper_figure_demo(
        self,
        num_keypoints: int = 10,
        target_idx: Optional[int] = None,
        hold_seconds: float = 0.0,
    ) -> dict:
        """Create a single illustrative online-expansion scene for the paper."""
        if self.visualizer is None:
            raise ValueError("Paper figure demo requires --visualize")

        print(f"\n{'=' * 60}")
        print("Preparing paper figure demo")
        print(f"{'=' * 60}")

        if self.warmstart and len(self.gcs_planner.regions) == 0:
            self._run_warmstart()

        assert self.visualizer is not None
        self.visualizer.clear_figure_artifacts()
        self.visualizer.update_robot(self.current_qpos)

        if target_idx is None:
            target_idx, target_qpos = self._pick_paper_figure_target()
        else:
            target_qpos = np.array(self.shelf_configs[target_idx])

        if target_qpos is None or target_idx is None:
            raise RuntimeError(
                "Could not find a target that requires online expansion. "
                "Try another scene or disable warmstart."
            )

        print(f"Selected target index: {target_idx}", flush=True)

        current_color = Rgba(1.0, 0.85, 0.15, 0.95)
        target_color = Rgba(0.92, 0.15, 0.20, 0.95)
        keypoint_color = Rgba(0.10, 0.45, 1.0, 0.95)
        added_region_color = Rgba(0.15, 0.80, 0.85, 0.90)
        pruned_region_color = Rgba(0.55, 0.55, 0.55, 0.75)

        self.visualizer.draw_config_marker(
            self.current_qpos,
            current_color,
            "figure/current",
            radius=0.03,
        )
        self.visualizer.draw_config_marker(
            target_qpos,
            target_color,
            "figure/target",
            radius=0.03,
        )

        goal_in_gcs = self.check_if_point_in_iris_regions(target_qpos)
        gcs_success_before = False
        if goal_in_gcs:
            try:
                gcs_success_before = self.gcs_planner.solve_from_configs(
                    self.current_qpos,
                    target_qpos,
                    build_missing_regions=False,
                )
            except Exception:
                gcs_success_before = False

        print(f"Target initially in GCS coverage: {goal_in_gcs}", flush=True)
        print(f"GCS solves before expansion:     {gcs_success_before}", flush=True)

        rrt_path = self.rrt_planner.plan_bidirectional(
            self.current_qpos, target_qpos, goal_tolerance=0.15
        )
        if rrt_path is None:
            rrt_path = self.rrt_planner.plan(self.current_qpos, target_qpos, goal_tolerance=0.15)
        if rrt_path is None:
            raise RuntimeError("RRT* failed to find a path for the paper figure demo.")

        self.visualizer.draw_path(rrt_path, use_gcs=False)

        keypoints = extract_keypoints_uniform(rrt_path, num_points=num_keypoints)
        self.visualizer.draw_config_markers(
            keypoints,
            group_name="figure/keypoints",
            color=keypoint_color,
            radius=0.018,
        )

        build_time = self.iris_region_builder.build_regions_from_seeds(
            keypoints,
            existing_regions=self.gcs_planner.regions,
        )

        new_regions = self.iris_region_builder.regions
        added_region_centers = []
        pruned_region_centers = []
        for region in new_regions:
            try:
                center = region.ChebyshevCenter()
            except Exception:
                continue
            if self._is_redundant_region(region):
                pruned_region_centers.append(center)
            else:
                added_region_centers.append(center)

        self.visualizer.draw_config_markers(
            added_region_centers,
            group_name="figure/added_region_centers",
            color=added_region_color,
            radius=0.014,
        )
        self.visualizer.draw_config_markers(
            pruned_region_centers,
            group_name="figure/pruned_region_centers",
            color=pruned_region_color,
            radius=0.012,
        )

        added = self.add_regions_to_gcs(new_regions)
        success_after = self.gcs_planner.solve_from_configs(
            self.current_qpos,
            target_qpos,
            build_missing_regions=False,
        )

        if success_after and self.gcs_planner.trajectory is not None:
            traj = self.gcs_planner.trajectory
            times = np.linspace(traj.start_time(), traj.end_time(), 25)
            path_samples = [traj.value(t).flatten() for t in times]
            self.visualizer.draw_path(path_samples, use_gcs=True)

        print(f"RRT path points:                 {len(rrt_path)}", flush=True)
        print(f"Uniform keypoints:               {len(keypoints)}", flush=True)
        print(f"IRIS build time:                 {build_time:.2f}s", flush=True)
        print(f"New regions added to GCS:        {added}", flush=True)
        print(f"Projected redundant regions:     {len(pruned_region_centers)}", flush=True)
        print(f"GCS solves after expansion:      {success_after}", flush=True)
        print(f"Meshcat URL: {self.visualizer.web_url()}", flush=True)
        print(f"{'=' * 60}\n")

        self._hold_figure_scene(hold_seconds)

        return {
            "target_idx": int(target_idx),
            "goal_in_gcs_before": bool(goal_in_gcs),
            "gcs_success_before": bool(gcs_success_before),
            "rrt_points": len(rrt_path),
            "keypoints": len(keypoints),
            "build_time": float(build_time),
            "regions_added": int(added),
            "redundant_regions": len(pruned_region_centers),
            "gcs_success_after": bool(success_after),
            "meshcat_url": self.visualizer.web_url(),
        }

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
        self.stats.total_regions = len(self.gcs_planner.regions)
        self.stats.total_regions_after_pruning = len(self.gcs_planner.regions)
        if self.logging:
            print(
                f"  [OnlineGCS] Added {added_count} new regions "
                f"(total: {len(self.gcs_planner.regions)})",
                flush=True,
            )
        return added_count

    def _is_redundant_region(self, new_region: HPolyhedron, containment_check: bool = True) -> bool:
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

    def get_random_point_from_shelf_configs(self) -> int:
        """Get a random target configuration from the shelf configs."""
        return choose_target_index(len(self.shelf_configs), self.current_idx, self.rng)

    def get_statistics(self) -> dict:
        """Return current statistics of the online GCS algorithm."""
        return {
            "warmstart_time": self.stats.warmstart_time,
            "warmstart_regions": self.stats.warmstart_regions,
            "regions_in_gcs": len(self.gcs_planner.regions),
            "total_regions": len(self.gcs_planner.regions),
            "gcs_success_count": self.stats.gcs_success_count,
            "rrt_fallback_count": self.stats.rrt_fallback_count,
            "total_queries": self.stats.total_queries,
            "gcs_success_rate": self.stats.gcs_success_count / max(1, self.stats.total_queries),
            "total_regions_added": self.stats.total_regions_added,
            "total_regions_after_pruning": len(self.gcs_planner.regions),
            "total_keypoints_requested": self.stats.total_keypoints_requested,
            "total_iris_regions_built": self.stats.total_iris_regions_built,
            "ways_rrt_time": {
                way: np.mean(self.stats.stats_per_each_query_rrt[way][1])
                for way in self.stats.stats_per_each_query_rrt
            },
            "ways_gcs_time": {
                way: np.mean(self.stats.stats_per_each_query_gcs[way][1])
                for way in self.stats.stats_per_each_query_gcs
            },
            "ways_gcs_vanilla_time": {
                way: np.mean(self.stats.stats_per_each_query_gcs[way][2])
                for way in self.stats.stats_per_each_query_gcs
            },
            "ways_rrt_path_length": {
                way: np.mean(self.stats.stats_per_each_query_rrt[way][0])
                for way in self.stats.stats_per_each_query_rrt
            },
            "ways_gcs_path_length": {
                way: np.mean(self.stats.stats_per_each_query_gcs[way][0])
                for way in self.stats.stats_per_each_query_gcs
            },
            "ways_iris_build_time": {
                way: np.mean(self.stats.stats_per_way_iris_build_time[way])
                for way in self.stats.stats_per_way_iris_build_time
            },
            "ways_trajopt_time": {
                way: np.mean(self.stats.stats_per_each_query_trajopt[way][1])
                for way in self.stats.stats_per_each_query_trajopt
            },
            "ways_trajopt_path_length": {
                way: np.mean(
                    [
                        path_length
                        for path_length, success in zip(
                            self.stats.stats_per_each_query_trajopt[way][0],
                            self.stats.stats_per_each_query_trajopt[way][2],
                        )
                        if success > 0.5
                    ]
                )
                if any(s > 0.5 for s in self.stats.stats_per_each_query_trajopt[way][2])
                else 0.0
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
        print(f"\n{'=' * 50}")
        print("Online GCS Statistics")
        print(f"{'=' * 50}")
        print(f"  Total queries:      {stats['total_queries']}")
        print(f"  GCS successes:      {stats['gcs_success_count']}")
        print(f"  RRT fallbacks:      {stats['rrt_fallback_count']}")
        print(f"  GCS success rate:   {stats['gcs_success_rate'] * 100:.1f}%")
        print(f"  Total regions:      {stats['total_regions']} (current graph)")
        print(f"  Regions added:      {stats['total_regions_added']} (accepted over this run)")
        print(f"  Keypoints requested:{stats['total_keypoints_requested']:>7}")
        print(f"  IRIS regions built: {stats['total_iris_regions_built']}")
        print(f"  Regions in GCS:     {stats['regions_in_gcs']} (current graph)")
        if stats["warmstart_regions"] > 0:
            print(
                f"  Warmstart regions:  {stats['warmstart_regions']} (in {stats['warmstart_time']:.1f}s)"
            )

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
                print(f"  Optimization success rate for {way}: {trajopt_success * 100:.0f}%")

            if iris_build_time is not None:
                print(f"  IRIS build time for {way}: {iris_build_time:.5f}s")
            else:
                print(f"  IRIS build time for {way}: 0.00s (regions from other paths)")

            # --- Trajectory quality metrics (arc-length-normalised) ---
            gcs_qlist = self.stats.stats_gcs_quality.get(way, [])
            topt_qlist = self.stats.stats_trajopt_quality.get(way, [])
            if gcs_qlist or topt_qlist:
                for metric_key, label in [
                    ("curv_max", "Max curvature"),
                    ("curv_integral", "Total curvature (∫κ ds)"),
                    ("torsion_max", "Max torsion"),
                    ("torsion_integral", "Total torsion"),
                ]:
                    gcs_val = np.mean([m[metric_key] for m in gcs_qlist]) if gcs_qlist else None
                    topt_val = np.mean([m[metric_key] for m in topt_qlist]) if topt_qlist else None
                    gcs_s = f"{gcs_val:.4f}" if gcs_val is not None else "N/A"
                    topt_s = f"{topt_val:.4f}" if topt_val is not None else "N/A"
                    print(f"  {label}: GCS={gcs_s}  Opt={topt_s}")

        print(f"{'=' * 50}\n")

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
        self.stats.total_regions = len(non_redundant)
        self.stats.total_regions_after_pruning = len(non_redundant)

        if self.logging and pruned_count > 0:
            print(f"  [OnlineGCS] Pruned {pruned_count} redundant regions", flush=True)

        return pruned_count

    # ==================== WARMSTART ====================

    def _farthest_point_seeds(
        self, candidates: List[np.ndarray], n_select: int, initial_seeds: List[np.ndarray]
    ) -> List[np.ndarray]:
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
                np.linalg.norm(cand[:, np.newaxis, :] - sel_arr[np.newaxis, :, :], axis=2),
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
        print(f"\n{'=' * 60}")
        print(f"Warmstart: building up to {self.warmstart_seeds} initial IRIS regions")
        print(f"{'=' * 60}")

        t0 = time.time()

        initial_seeds: List[np.ndarray] = []
        for cfg in self.shelf_configs:
            q = np.array(cfg)
            if self.iris_region_builder.is_collision_free(q):
                initial_seeds.append(q)

        if self.logging:
            print(f"  {len(initial_seeds)} shelf configs as initial seeds", flush=True)

        n_candidates = max(500, self.warmstart_seeds * 50)
        candidates = self.iris_region_builder.sample_collision_free(n_candidates, self.rng)
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
            f"Warmstart done: {added} regions in {total_time:.1f}s (IRIS build: {build_time:.1f}s)",
            flush=True,
        )
        print(f"{'=' * 60}\n")
        return added, total_time

    # ==================== MAIN LOOP ====================

    def run(self, num_keypoints: int = 10, prune_interval: int = 20, animation_speed: float = 1.0):
        """
        Run the online GCS algorithm.

        Args:
            num_keypoints: Number of keypoints to extract from RRT path
            prune_interval: How often to prune redundant regions (0 = never)
            animation_speed: Speed multiplier for animations (higher = faster)
        """
        print(f"\n{'=' * 60}")
        print(f"Starting Online GCS for {self.scene_type.name}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Initial position: [{', '.join(f'{x:.2f}' for x in self.current_qpos[:3])}...]")
        if self.visualizer is not None:
            print("Visualization: ENABLED")
            print(f"Meshcat URL: {self.visualizer.web_url()}")
        print(f"{'=' * 60}\n")

        if self.warmstart:
            self._run_warmstart()

        if self.visualizer is not None:
            self.visualizer.update_robot(self.current_qpos)
            time.sleep(0.5)

        try:
            overall_time = time.time()
            gcs_opt_time = 0.0
            all_iris_build_time = 0
            opt_success_count = 0

            for iteration in range(self.max_iterations):
                # Harvest any regions built by parallel workers
                if self.exploration_coordinator is not None:
                    parallel_regions = self.exploration_coordinator.collect_results(
                        self.gcs_planner.regions, self.rng
                    )
                    if parallel_regions:
                        added = self.add_regions_to_gcs(parallel_regions)
                        if self.logging and added > 0:
                            print(
                                f"  [Parallel] Integrated {added} regions from workers", flush=True
                            )

                self.stats.total_queries += 1

                target_idx = self.get_random_point_from_shelf_configs()
                target_qpos = np.array(self.shelf_configs[target_idx])

                if self.logging:
                    print(
                        f"\n[Iter {iteration + 1}/{self.max_iterations}] "
                        f"Target: [{', '.join(f'{x:.2f}' for x in target_qpos[:3])}...]"
                    )

                if self.visualizer is not None:
                    self.visualizer.draw_target(target_qpos)

                current_way_name = f"{self.current_idx}-{target_idx}"
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
                        opt_success_count += 1
                        topt_qm = compute_trajectory_metrics(trajopt_result.trajectory)
                        self.stats.stats_trajopt_quality.setdefault(current_way_name, []).append(
                            topt_qm
                        )
                    if (
                        self.visualizer is not None
                        and trajopt_result.success
                        and trajopt_result.path
                    ):
                        trajopt_color = Rgba(0.8, 0.2, 0.8, 0.7)
                        self.visualizer.draw_path(trajopt_result.path, color=trajopt_color)
                    if self.logging:
                        if trajopt_result.success:
                            print(
                                f"  TrajOpt path length: {trajopt_result.path_length:.3f} "
                                f"(was {trajopt_result.initial_path_length:.3f}), "
                                f"time: {trajopt_result.solve_time:.3f}s"
                            )
                        else:
                            print(f"  TrajOpt FAILED, time: {trajopt_result.solve_time:.3f}s")

                # start_in_gcs = self.check_if_point_in_iris_regions(self.current_qpos)
                # ================== end opt baseline ================
                goal_in_gcs = self.check_if_point_in_iris_regions(target_qpos)

                if goal_in_gcs:
                    if self.logging:
                        print("  Both points in GCS, trying GCS planning...")

                    gcs_start_time = time.time()
                    success = self.gcs_planner.solve_from_configs(
                        self.current_qpos,
                        target_qpos,
                        build_missing_regions=True,  ## !! changed to true for testing
                    )
                    gcs_time = time.time() - gcs_start_time
                    # Vanilla GCS solve time is stored on the planner.
                    gcs_opt_time += gcs_time
                    gcs_vanilla_time = self.gcs_planner.solve_time

                    if success:
                        self.stats.gcs_success_count += 1

                        if self.visualizer is not None and self.gcs_planner.trajectory is not None:
                            traj = self.gcs_planner.trajectory
                            times = np.linspace(traj.start_time(), traj.end_time(), 20)
                            path_samples = [traj.value(t).flatten() for t in times]
                            self.visualizer.draw_path(path_samples, use_gcs=True)
                            self.visualizer.animate_trajectory(traj, duration=2.0 / animation_speed)

                        current_way_name = f"{self.current_idx}-{target_idx}"
                        if current_way_name not in self.stats.stats_per_each_query_gcs:
                            self.stats.stats_per_each_query_gcs[current_way_name] = [[], [], []]
                        self.stats.stats_per_each_query_gcs[current_way_name][0].append(
                            self.gcs_planner.path_length
                        )
                        self.stats.stats_per_each_query_gcs[current_way_name][1].append(gcs_time)
                        self.stats.stats_per_each_query_gcs[current_way_name][2].append(
                            gcs_vanilla_time
                        )
                        if current_way_name not in self.stats.stats_per_way_iris_build_time:
                            self.stats.stats_per_way_iris_build_time[current_way_name] = []
                        self.stats.stats_per_way_iris_build_time[current_way_name].append(0.0)
                        if self.gcs_planner.trajectory is not None:
                            gcs_qm = compute_trajectory_metrics(self.gcs_planner.trajectory)
                            self.stats.stats_gcs_quality.setdefault(current_way_name, []).append(
                                gcs_qm
                            )

                        self.current_qpos = target_qpos.copy()
                        self.current_idx = target_idx
                        if self.logging:
                            print(f"  GCS SUCCESS! Path length: {self.gcs_planner.path_length:.3f}")
                        continue
                    else:
                        if self.logging:
                            print("  GCS failed (regions not connected), falling back to RRT")

                self.stats.rrt_fallback_count += 1

                if self.logging:
                    print("  Using RRT* path from current to target...")

                rrt_path = self.rrt_planner.plan_bidirectional(
                    self.current_qpos, target_qpos, goal_tolerance=0.15
                )

                if rrt_path is not None:
                    if self.logging:
                        print(
                            f"  RRT found path with {len(rrt_path)} points, "
                            f"length: {self.rrt_planner.path_length:.3f}"
                        )

                    if self.visualizer is not None:
                        self.visualizer.draw_path(rrt_path, use_gcs=False)

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
                    all_iris_build_time += build_time
                    new_regions = self.iris_region_builder.regions
                    added = self.add_regions_to_gcs(new_regions)

                    if self.logging:
                        print(f"  Added {added} regions in {build_time:.2f}s")

                    # Submit exploration tasks for workers to build regions in uncovered space
                    if self.exploration_coordinator is not None:
                        q_lower = self.iris_region_builder.q_lower
                        q_upper = self.iris_region_builder.q_upper
                        seeds = self.exploration_coordinator.generate_exploration_seeds(
                            self.gcs_planner.regions,
                            q_lower,
                            q_upper,
                            self.rng,
                            count=min(4, self.num_workers * 2),
                        )
                        if seeds:
                            self.exploration_coordinator.submit_tasks(seeds, interval_id=iteration)
                            if self.logging:
                                print(
                                    f"  [Parallel] Submitted {len(seeds)} exploration tasks",
                                    flush=True,
                                )

                    gcs_start_time = time.time()
                    success = self.gcs_planner.solve_from_configs(
                        self.current_qpos, target_qpos, build_missing_regions=True
                    )
                    gcs_time = time.time() - gcs_start_time
                    gcs_vanilla_time = self.gcs_planner.solve_time

                    if success:
                        self.stats.gcs_success_count += 1
                        if self.visualizer is not None and self.gcs_planner.trajectory is not None:
                            traj = self.gcs_planner.trajectory
                            times = np.linspace(traj.start_time(), traj.end_time(), 20)
                            path_samples = [traj.value(t).flatten() for t in times]
                            self.visualizer.draw_path(path_samples, use_gcs=True)
                            self.visualizer.animate_trajectory(
                                self.gcs_planner.trajectory, duration=2.0 / animation_speed
                            )

                        if current_way_name not in self.stats.stats_per_each_query_gcs:
                            self.stats.stats_per_each_query_gcs[current_way_name] = [[], [], []]
                        self.stats.stats_per_each_query_gcs[current_way_name][0].append(
                            self.gcs_planner.path_length
                        )
                        self.stats.stats_per_each_query_gcs[current_way_name][1].append(gcs_time)
                        self.stats.stats_per_each_query_gcs[current_way_name][2].append(
                            gcs_vanilla_time
                        )

                        if self.gcs_planner.trajectory is not None:
                            gcs_qm = compute_trajectory_metrics(self.gcs_planner.trajectory)
                            self.stats.stats_gcs_quality.setdefault(current_way_name, []).append(
                                gcs_qm
                            )
                        self.current_qpos = target_qpos.copy()
                        self.current_idx = target_idx
                        if self.logging:
                            print(
                                f"  GCS with new regions SUCCESS! "
                                f"Path length: {self.gcs_planner.path_length:.3f}"
                            )
                    else:
                        if self.visualizer is not None:
                            self.visualizer.animate_path(rrt_path, duration=2.0 / animation_speed)
                        # RRT already recorded at start of iteration

                        self.current_qpos = target_qpos.copy()
                        self.current_idx = target_idx
                        if self.logging:
                            print("  GCS still failed, using RRT path to reach target")
                else:
                    print("  [WARNING] RRT failed to find path from current to target")
                    continue

                if self.visualizer is not None:
                    self.visualizer.update_robot(self.current_qpos)

                if prune_interval > 0 and (iteration + 1) % prune_interval == 0:
                    pruned = self.prune_redundant_regions()
                    if self.logging and pruned > 0:
                        print(f"  Pruned {pruned} redundant regions")

                if (iteration + 1) % 10 == 0:
                    print(
                        f"[Progress] Iteration {iteration + 1}/{self.max_iterations}, "
                        f"Regions: {len(self.gcs_planner.regions)}, "
                        f"GCS rate: {self.stats.gcs_success_count}/{self.stats.total_queries} "
                        f"({self.stats.gcs_success_count / self.stats.total_queries * 100:.1f}%)"
                    )
            overall_time = time.time() - overall_time
            print(f"########### Overall time: {overall_time:.2f}s")
            print(f"########### GCS opt. time: {gcs_opt_time:.2f}s")
            print(f"########### IRIS build time: {all_iris_build_time:.2f}s")
            print(
                f"########### Opt success rate: {opt_success_count / self.stats.total_queries * 100:.1f}%"
            )
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
        if self.visualizer is not None:
            print("Visualization: ENABLED")
            print(f"Meshcat URL: {self.visualizer.web_url()}")
        print(f"{'=' * 60}\n")

        if self.visualizer is not None:
            self.visualizer.update_robot(self.current_qpos)
            time.sleep(0.5)

        try:
            overall_time = time.time()
            for iteration in range(self.max_iterations):
                self.stats.total_queries += 1

                target_idx = self.get_random_point_from_shelf_configs()
                target_qpos = np.array(self.shelf_configs[target_idx])

                if self.logging:
                    print(
                        f"\n[Iter {iteration + 1}/{self.max_iterations}] "
                        f"Target: [{', '.join(f'{x:.2f}' for x in target_qpos[:3])}...]"
                    )

                if self.visualizer is not None:
                    self.visualizer.draw_target(target_qpos)

                rrt_path = self.rrt_planner.plan_bidirectional(
                    self.current_qpos, target_qpos, goal_tolerance=0.15
                )

                if rrt_path is not None:
                    print(iteration)

                if self.visualizer is not None:
                    self.visualizer.draw_path(rrt_path, use_gcs=False)

            overall_time = time.time() - overall_time
            print(f"########### Overall time: {overall_time:.2f}s")
        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Stopping online GCS...")

    def run_opt_only(self):
        if self.visualizer is not None:
            print("Visualization: ENABLED")
            print(f"Meshcat URL: {self.visualizer.web_url()}")
        print(f"{'=' * 60}\n")

        if self.visualizer is not None:
            self.visualizer.update_robot(self.current_qpos)
            time.sleep(0.5)

        try:
            overall_time = time.time()
            opt_success_count = 0
            for iteration in range(self.max_iterations):
                self.stats.total_queries += 1

                target_idx = self.get_random_point_from_shelf_configs()
                target_qpos = np.array(self.shelf_configs[target_idx])

                if self.logging:
                    print(
                        f"\n[Iter {iteration + 1}/{self.max_iterations}] "
                        f"Target: [{', '.join(f'{x:.2f}' for x in target_qpos[:3])}...]"
                    )

                if self.visualizer is not None:
                    self.visualizer.draw_target(target_qpos)

                rrt_path = self.rrt_planner.plan_bidirectional(
                    self.current_qpos, target_qpos, goal_tolerance=0.15
                )

                if rrt_path is not None and len(rrt_path) >= 2:
                    trajopt_result = self.trajopt_solver.optimize_rrt_path(rrt_path)
                    if trajopt_result.success:
                        opt_success_count += 1
                    if (
                        self.visualizer is not None
                        and trajopt_result.success
                        and trajopt_result.path
                    ):
                        trajopt_color = Rgba(0.8, 0.2, 0.8, 0.7)
                        self.visualizer.draw_path(trajopt_result.path, color=trajopt_color)
                    if self.logging:
                        if trajopt_result.success:
                            print(
                                f"  TrajOpt path length: {trajopt_result.path_length:.3f} "
                                f"(was {trajopt_result.initial_path_length:.3f}), "
                                f"time: {trajopt_result.solve_time:.3f}s"
                            )
                        else:
                            print(f"  TrajOpt FAILED, time: {trajopt_result.solve_time:.3f}s")

                if self.visualizer is not None:
                    self.visualizer.draw_path(rrt_path, use_gcs=False)

            overall_time = time.time() - overall_time
            print(f"########### Overall time: {overall_time:.2f}s")
            print(
                f"########### Opt success rate: {opt_success_count / self.stats.total_queries * 100:.1f}%"
            )
        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Stopping online GCS...")
