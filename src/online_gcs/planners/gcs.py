import heapq
import time
from collections import deque
from typing import List, Optional, Set, Tuple

import numpy as np

from pydrake.all import (
    GcsTrajectoryOptimization,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    IrisOptions,
    IrisNp,
    LoadIrisRegionsYamlFile,
    CompositeTrajectory,
)
from pydrake.geometry.optimization import Point

from online_gcs.regions.iris import IRISRegionBuilder
from online_gcs.scenes import SceneBuilder, SceneType
from online_gcs.utils import solve_IK


class GCSPathPlanner:
    def __init__(
        self,
        regions: List[HPolyhedron],
        plant,
        plant_context,
        gripper_frame=None,
        k_shortest_paths: int = 1,
    ):
        self.plant = plant
        self.plant_context = plant_context
        self.regions = list(regions)
        self.nq = plant.num_positions()
        self.gripper_frame = gripper_frame
        self.k_shortest_paths = max(1, k_shortest_paths)

        if self.gripper_frame is None:
            try:
                wsg = self.plant.GetModelInstanceByName("gripper")
                self.gripper_frame = self.plant.GetFrameByName("body", wsg)
            except BaseException:
                pass

        self.trajectory: Optional[CompositeTrajectory] = None
        self.solve_time: float = 0.0
        self.path_length: float = 0.0
        self.success: bool = False

    @classmethod
    def from_iris_builder(
        cls, iris_builder: IRISRegionBuilder, k_shortest_paths: int = 1
    ) -> "GCSPathPlanner":
        return cls(
            regions=iris_builder.regions,
            plant=iris_builder.plant,
            plant_context=iris_builder.plant_context,
            k_shortest_paths=k_shortest_paths,
        )

    @classmethod
    def from_yaml(
        cls, yaml_path: str, scene_type: SceneType, k_shortest_paths: int = 1
    ) -> "GCSPathPlanner":
        diagram = SceneBuilder.build(scene_type)
        context = diagram.CreateDefaultContext()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyContextFromRoot(context)

        regions_dict = LoadIrisRegionsYamlFile(yaml_path)
        regions = list(regions_dict.values())

        return cls(
            regions=regions,
            plant=plant,
            plant_context=plant_context,
            k_shortest_paths=k_shortest_paths,
        )

    def _point_to_config(
        self, point_3d: np.ndarray, q_initial: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        if self.gripper_frame is None:
            print("WARNING: No gripper frame found, cannot solve IK", flush=True)
            return None

        if q_initial is None:
            q_initial = np.zeros(self.nq)

        q_solution = solve_IK(
            self.plant,
            self.plant_context,
            goal_pos=list(point_3d),
            ee_frame=self.gripper_frame,
            q_initial=q_initial,
        )
        return q_solution

    def _find_containing_region(self, q: np.ndarray) -> int:
        for i, region in enumerate(self.regions):
            if region.PointInSet(q):
                return i
        return -1

    def _build_region_around(
        self, q: np.ndarray, collision_samples: int = 3
    ) -> Optional[HPolyhedron]:
        self.plant.SetPositions(self.plant_context, q)

        opts = IrisOptions()
        opts.num_collision_infeasible_samples = collision_samples
        opts.random_seed = 12345
        opts.require_sample_point_is_contained = True

        try:
            region = IrisNp(self.plant, self.plant_context, opts)
            return region
        except Exception:
            try:
                opts = IrisOptions()
                opts.num_collision_infeasible_samples = 10
                opts.configuration_space_margin = 1e-4
                opts.random_seed = 12345
                opts.require_sample_point_is_contained = True
                region = IrisNp(self.plant, self.plant_context, opts)
                return region
            except Exception as e:
                print(f"WARNING: Failed to build IRIS region: {e}", flush=True)
                return None

    def _ensure_point_in_region(
        self, q: np.ndarray, point_name: str, build_if_missing: bool = True
    ) -> int:
        region_idx = self._find_containing_region(q)

        if region_idx >= 0:
            print(f"  {point_name} found in region {region_idx}", flush=True)
            return region_idx

        if not build_if_missing:
            print(f"WARNING: {point_name} not in any region (skipping IRIS build)", flush=True)
            return -1

        print(
            f"WARNING: {point_name} not in any existing region, building new one (this is SLOW)...",
            flush=True,
        )
        new_region = self._build_region_around(q)

        if new_region is None:
            print(f"ERROR: Could not build region around {point_name}", flush=True)
            return -1

        self.regions.append(new_region)
        return len(self.regions) - 1

    def _compute_path_length(self, traj: CompositeTrajectory, num_samples: int = 500) -> float:
        times = np.linspace(traj.start_time(), traj.end_time(), num_samples)
        positions = traj.vector_values(times)
        return np.sum(np.linalg.norm(np.diff(positions, axis=1), axis=0))

    def solve(
        self,
        start_3d: np.ndarray,
        goal_3d: np.ndarray,
        order: int = 3,
        max_rounded_paths: int = 10,
        build_missing_regions: bool = True,
    ) -> bool:
        """
        Solve path planning from start_3d to goal_3d.

        Args:
            start_3d: Start position in 3D (end-effector)
            goal_3d: Goal position in 3D (end-effector)
            order: Bezier curve order (1=fast, 3=smooth/short)
            max_rounded_paths: GCS rounding parameter
            build_missing_regions: If True, build IRIS for points not in regions (SLOW!)

        Returns:
            True if path found, False otherwise
        """
        self.success = False
        self.trajectory = None
        self.path_length = 0.0

        q_start = self._point_to_config(start_3d)
        if q_start is None:
            print("ERROR: IK failed for start point", flush=True)
            return False

        q_goal = self._point_to_config(goal_3d, q_initial=q_start)
        if q_goal is None:
            print("ERROR: IK failed for goal point", flush=True)
            return False

        start_region_idx = self._ensure_point_in_region(q_start, "start", build_missing_regions)
        goal_region_idx = self._ensure_point_in_region(q_goal, "goal", build_missing_regions)

        if start_region_idx < 0 or goal_region_idx < 0:
            print(
                "TIP: Try different start/goal points that are covered by existing regions",
                flush=True,
            )
            return False

        print(f"Start in region {start_region_idx}, goal in region {goal_region_idx}", flush=True)

        connected_regions = self._get_connected_subgraph(start_region_idx, goal_region_idx)
        if connected_regions is None:
            print("ERROR: No path exists between start and goal regions", flush=True)
            return False

        print(
            f"Using {len(connected_regions)} connected regions (of {len(self.regions)} total)",
            flush=True,
        )

        return self._solve_gcs_internal(
            [self.regions[i] for i in connected_regions], q_start, q_goal, order, max_rounded_paths
        )

    def _get_connected_subgraph(
        self,
        start_idx: int,
        goal_idx: int,
        start_nearest_idx: int = None,
        goal_nearest_idx: int = None,
    ) -> Optional[List[int]]:
        n = len(self.regions)
        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.regions[i].IntersectsWith(self.regions[j]):
                    adj[i].append(j)
                    adj[j].append(i)

        if not adj[start_idx] and start_nearest_idx is not None:
            adj[start_idx].append(start_nearest_idx)
            adj[start_nearest_idx].append(start_idx)
        if not adj[goal_idx] and goal_nearest_idx is not None:
            adj[goal_idx].append(goal_nearest_idx)
            adj[goal_nearest_idx].append(goal_idx)

        paths = self._yen_k_shortest_paths(adj, start_idx, goal_idx, self.k_shortest_paths)
        if not paths:
            return None

        region_set: Set[int] = set()
        for path in paths:
            region_set.update(path)
        return sorted(region_set)

    # ------------------------------------------------------------------
    #  Yen's K-shortest loopless paths  (unweighted graph via BFS)
    # ------------------------------------------------------------------

    @staticmethod
    def _bfs_shortest_path(
        adj: List[List[int]],
        source: int,
        target: int,
        removed_edges: Set[Tuple[int, int]] = None,
        removed_nodes: Set[int] = None,
    ) -> Optional[List[int]]:
        if removed_edges is None:
            removed_edges = set()
        if removed_nodes is None:
            removed_nodes = set()
        if source in removed_nodes or target in removed_nodes:
            return None

        visited = {source}
        queue = deque([source])
        parent = {source: -1}

        while queue:
            curr = queue.popleft()
            if curr == target:
                path: List[int] = []
                node = target
                while node != -1:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for neighbor in adj[curr]:
                if neighbor in visited or neighbor in removed_nodes:
                    continue
                edge = (min(curr, neighbor), max(curr, neighbor))
                if edge in removed_edges:
                    continue
                visited.add(neighbor)
                parent[neighbor] = curr
                queue.append(neighbor)

        return None

    @classmethod
    def _yen_k_shortest_paths(
        cls, adj: List[List[int]], source: int, target: int, k: int
    ) -> List[List[int]]:
        first_path = cls._bfs_shortest_path(adj, source, target)
        if first_path is None:
            return []
        if k <= 1:
            return [first_path]

        A: List[List[int]] = [first_path]
        B: List[Tuple[int, int, List[int]]] = []
        counter = 0

        for k_i in range(1, k):
            prev_path = A[k_i - 1]
            for i in range(len(prev_path) - 1):
                spur_node = prev_path[i]
                root_path = prev_path[: i + 1]

                removed_edges: Set[Tuple[int, int]] = set()
                for p in A:
                    if len(p) > i and p[: i + 1] == root_path:
                        edge = (min(p[i], p[i + 1]), max(p[i], p[i + 1]))
                        removed_edges.add(edge)

                removed_nodes: Set[int] = set(root_path[:-1])

                spur_path = cls._bfs_shortest_path(
                    adj, spur_node, target, removed_edges, removed_nodes
                )
                if spur_path is not None:
                    candidate = root_path[:-1] + spur_path
                    is_dup = candidate in A
                    if not is_dup:
                        for _, _, bp in B:
                            if bp == candidate:
                                is_dup = True
                                break
                    if not is_dup:
                        heapq.heappush(B, (len(candidate), counter, candidate))
                        counter += 1

            if not B:
                break
            _, _, best = heapq.heappop(B)
            A.append(best)

        return A

    def _solve_gcs_internal(
        self,
        regions: List[HPolyhedron],
        q_start: np.ndarray,
        q_goal: np.ndarray,
        order: int,
        max_rounded_paths: int,
        start_nearest_idx: int = None,
        goal_nearest_idx: int = None,
    ) -> bool:
        n = len(regions)

        gcs = GcsTrajectoryOptimization(self.nq)

        region_subgraphs = []
        for i, r in enumerate(regions):
            sg = gcs.AddRegions([r], order=order, name=f"r{i}")
            region_subgraphs.append(sg)

        edges_added = 0
        for i in range(n):
            for j in range(i + 1, n):
                if regions[i].IntersectsWith(regions[j]):
                    gcs.AddEdges(region_subgraphs[i], region_subgraphs[j])
                    gcs.AddEdges(region_subgraphs[j], region_subgraphs[i])
                    edges_added += 1

        source = gcs.AddRegions([Point(q_start)], order=0, name="source")
        target = gcs.AddRegions([Point(q_goal)], order=0, name="target")

        for i, r in enumerate(regions):
            if r.PointInSet(q_start):
                gcs.AddEdges(source, region_subgraphs[i])
                if start_nearest_idx is not None:
                    gcs.AddEdges(region_subgraphs[i], region_subgraphs[start_nearest_idx])
                print(f"  Source -> region {i}", flush=True)

        for i, r in enumerate(regions):
            if r.PointInSet(q_goal):
                gcs.AddEdges(region_subgraphs[i], target)
                if goal_nearest_idx is not None:
                    gcs.AddEdges(region_subgraphs[goal_nearest_idx], region_subgraphs[i])
                print(f"  Region {i} -> target", flush=True)

        gcs.AddPathLengthCost()
        gcs.AddPathEnergyCost()
        gcs.AddTimeCost(weight=0.01)
        gcs.AddVelocityBounds(
            self.plant.GetVelocityLowerLimits(), self.plant.GetVelocityUpperLimits()
        )

        options = GraphOfConvexSetsOptions()
        options.preprocessing = True
        options.max_rounded_paths = max_rounded_paths

        t0 = time.perf_counter()

        try:
            traj, result = gcs.SolvePath(source, target, options)
            self.solve_time = time.perf_counter() - t0

            if result.is_success() and traj is not None:
                self.trajectory = traj
                self.path_length = self._compute_path_length(traj)
                self.success = True
                print(
                    f"SUCCESS! Time: {self.solve_time:.5f}s, Path length: {self.path_length:.3f}",
                    flush=True,
                )
            else:
                print(f"FAILED: No path found (time: {self.solve_time:.2f}s)", flush=True)

        except Exception as e:
            self.solve_time = time.perf_counter() - t0
            print(f"ERROR: {e}", flush=True)

        return self.success

    def solve_from_configs(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        order: int = 3,
        max_rounded_paths: int = 10,
        build_missing_regions: bool = False,
        start_nearest_idx: int = None,
        goal_nearest_idx: int = None,
    ) -> bool:
        """
        Solve path planning directly from joint configurations.

        Args:
            q_start: Start configuration in joint space
            q_goal: Goal configuration in joint space
            order: Bezier order (1=fast, 3=smooth/short)
            max_rounded_paths: GCS rounding parameter
            build_missing_regions: Build IRIS if points not in regions (SLOW!)
        """
        self.success = False
        self.trajectory = None
        self.path_length = 0.0

        start_region_idx = self._ensure_point_in_region(q_start, "start", build_missing_regions)
        goal_region_idx = self._ensure_point_in_region(q_goal, "goal", build_missing_regions)

        if start_region_idx < 0 or goal_region_idx < 0:
            print("TIP: Use configs that are inside existing IRIS regions", flush=True)
            return False

        print(f"Start in region {start_region_idx}, goal in region {goal_region_idx}", flush=True)

        connected_regions = self._get_connected_subgraph(
            start_region_idx, goal_region_idx, start_nearest_idx, goal_nearest_idx
        )
        if connected_regions is None:
            print("ERROR: No path exists between start and goal regions", flush=True)
            return False

        print(f"Using {len(connected_regions)} connected regions", flush=True)

        ind_start = (
            connected_regions.index(start_nearest_idx)
            if start_nearest_idx in connected_regions
            else None
        )
        ind_goal = (
            connected_regions.index(goal_nearest_idx)
            if goal_nearest_idx in connected_regions
            else None
        )

        print(f"Start nearest index: {ind_start}, goal nearest index: {ind_goal}", flush=True)

        return self._solve_gcs_internal(
            [self.regions[i] for i in connected_regions],
            q_start,
            q_goal,
            order,
            max_rounded_paths,
            start_nearest_idx=ind_start,
            goal_nearest_idx=ind_goal,
        )
