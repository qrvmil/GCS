import time
import numpy as np
from typing import List, Dict, Optional

from experiments.scene_types import SceneType
from helpers.scene_builder import SceneBuilder

from pydrake.all import (
    HPolyhedron,
    IrisOptions,
    IrisNp,
    SaveIrisRegionsYamlFile,
    LoadIrisRegionsYamlFile,
    InverseKinematics,
    Solve,
)
from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint


class IRISRegionBuilder:
    
    def __init__(self, scene_type: SceneType, random_seed: int = 42):
        self.scene_type = scene_type
        self.random_seed = random_seed
        
        self.diagram = SceneBuilder.build(scene_type)
        self.context = self.diagram.CreateDefaultContext()
        self.plant = self.diagram.GetSubsystemByName("plant")
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.scene_graph = self.diagram.GetSubsystemByName("scene_graph")
        
        self.regions: List[HPolyhedron] = []
        self.seed_configs: List[np.ndarray] = []
        self.build_time: float = 0.0
        
        self.q_lower = self.plant.GetPositionLowerLimits()
        self.q_upper = self.plant.GetPositionUpperLimits()
        self.nq = self.plant.num_positions()
    
    def is_collision_free(self, q: np.ndarray) -> bool:
        self.plant.SetPositions(self.plant_context, q)
        sg_context = self.scene_graph.GetMyContextFromRoot(self.context)
        query = self.scene_graph.get_query_output_port().Eval(sg_context)
        return not query.HasCollisions()
    
    def sample_collision_free(self, n: int, rng: np.random.Generator) -> List[np.ndarray]:
        configs = []
        for _ in range(n * 50):
            q = rng.uniform(self.q_lower, self.q_upper)
            if self.is_collision_free(q):
                configs.append(q.copy())
                if len(configs) >= n:
                    break
        return configs
    
    def build_regions(self, num_seeds: int, collision_samples: int = 3, 
                      rng: Optional[np.random.Generator] = None) -> float:
        if rng is None:
            rng = np.random.default_rng(self.random_seed)
        
        self.seed_configs = self.sample_collision_free(num_seeds, rng)
        self.regions = []
        self.build_time = 0.0
        
        for i, q in enumerate(self.seed_configs):
            self.plant.SetPositions(self.plant_context, q)
            
            opts = IrisOptions()
            opts.num_collision_infeasible_samples = collision_samples
            opts.random_seed = self.random_seed + i
            opts.require_sample_point_is_contained = True
            
            t0 = time.perf_counter()
            try:
                region = IrisNp(self.plant, self.plant_context, opts)
                self.regions.append(region)
            except Exception:
                pass
            self.build_time += time.perf_counter() - t0
        
        return self.build_time
    
    def count_isolated(self) -> int:
        n = len(self.regions)
        if n == 0:
            return 0
        
        isolated = 0
        for i in range(n):
            has_neighbor = False
            for j in range(n):
                if i != j and self.regions[i].IntersectsWith(self.regions[j]):
                    has_neighbor = True
                    break
            if not has_neighbor:
                isolated += 1
        return isolated
    
    def estimate_coverage(self, num_samples: int = 200, 
                          rng: Optional[np.random.Generator] = None) -> float:
        if not self.regions:
            return 0.0
        
        if rng is None:
            rng = np.random.default_rng(self.random_seed + 10000)
        
        test_points = self.sample_collision_free(num_samples, rng)
        if not test_points:
            return 0.0
        
        covered = 0
        for q in test_points:
            for region in self.regions:
                if region.PointInSet(q):
                    covered += 1
                    break
        return covered / len(test_points)
    
    def get_regions_dict(self) -> Dict[str, HPolyhedron]:
        return {f"region_{i}": r for i, r in enumerate(self.regions)}
    
    def save_regions(self, filepath: str):
        SaveIrisRegionsYamlFile(filepath, self.get_regions_dict())
    
    def load_regions(self, filepath: str):
        regions_dict = LoadIrisRegionsYamlFile(filepath)
        self.regions = list(regions_dict.values())
    
    @property
    def num_regions(self) -> int:
        return len(self.regions)
    
    @property
    def isolation_rate(self) -> float:
        if not self.regions:
            return 1.0
        return self.count_isolated() / len(self.regions)
    
    def _get_shelf_positions(self) -> List[np.ndarray]:
        match self.scene_type:
            case SceneType.SINGLE_SHELF:
                return [
                    np.array([0.70, 0.0, 0.30]),
                    np.array([0.70, 0.0, 0.50]),
                    np.array([0.70, 0.0, 0.70]),
                    np.array([0.65, 0.1, 0.40]),
                    np.array([0.65, -0.1, 0.60]),
                ]
            case SceneType.TWO_SHELVES:
                return [
                    np.array([0.75, -0.35, 0.45]),
                    np.array([0.75, 0.35, 0.45]),
                    np.array([0.75, -0.35, 0.65]),
                    np.array([0.75, 0.35, 0.65]),
                    np.array([0.70, -0.35, 0.55]),
                    np.array([0.70, 0.35, 0.55]),
                ]
            case SceneType.TABLE_THREE_SHELVES:
                return [
                    np.array([0.65, 0.50, 0.45]),
                    np.array([0.65, -0.50, 0.45]),
                    np.array([-0.75, 0.0, 0.45]),
                    np.array([0.60, 0.45, 0.35]),
                    np.array([0.60, -0.45, 0.35]),
                    np.array([-0.70, 0.1, 0.35]),
                ]
            
    def _solve_ik(self, goal_pos: np.ndarray, q_initial: np.ndarray) -> Optional[np.ndarray]:
        try:
            wsg = self.plant.GetModelInstanceByName("gripper")
            gripper_frame = self.plant.GetFrameByName("body", wsg)
        except:
            return None
        
        self.plant.SetPositions(self.plant_context, q_initial)
        
        ik = InverseKinematics(self.plant, self.plant_context)
        q = ik.q()
        prog = ik.prog()
        
        # Collision avoidance
        distance_constraint = MinimumDistanceLowerBoundConstraint(
            plant=self.plant, bound=0.01, plant_context=self.plant_context,
            influence_distance_offset=0.01
        )
        prog.AddConstraint(distance_constraint, q)
        
        # Position constraint
        p_BQ = np.array([0.0, 0.1, 0.0])
        ik.AddPositionConstraint(
            gripper_frame, p_BQ, self.plant.world_frame(),
            goal_pos, goal_pos
        )
        
        Q = np.eye(len(q_initial))
        prog.AddQuadraticErrorCost(Q, q_initial, q)
        prog.SetInitialGuess(q, q_initial)
        
        result = Solve(prog)
        if not result.is_success():
            return None
        
        return result.GetSolution(q)
    
    def sample_shelf_configs(self, rng: np.random.Generator) -> List[np.ndarray]:
        shelf_positions = self._get_shelf_positions()
        configs = []
        q_initial = np.zeros(self.nq)
        
        for pos in shelf_positions:
            for _ in range(3):
                perturbed_pos = pos + rng.uniform(-0.05, 0.05, 3)
                q_sol = self._solve_ik(perturbed_pos, q_initial)
                if q_sol is not None and self.is_collision_free(q_sol):
                    configs.append(q_sol)
                    q_initial = q_sol
                    break
        
        return configs
    
    def build_regions_mixed(self, num_seeds: int, shelf_ratio: float = 0.3,
                           collision_samples: int = 3,
                           rng: Optional[np.random.Generator] = None) -> float:
        if rng is None:
            rng = np.random.default_rng(self.random_seed)
        
        num_shelf = int(num_seeds * shelf_ratio)
        num_random = num_seeds - num_shelf
        
        shelf_configs = self.sample_shelf_configs(rng)
        
        if len(shelf_configs) < num_shelf:
            num_random += (num_shelf - len(shelf_configs))
        
        random_configs = self.sample_collision_free(num_random, rng)
        self.seed_configs = shelf_configs + random_configs
        
        self.regions = []
        self.build_time = 0.0
        
        for i, q in enumerate(self.seed_configs):
            self.plant.SetPositions(self.plant_context, q)
            
            opts = IrisOptions()
            opts.num_collision_infeasible_samples = collision_samples
            opts.random_seed = self.random_seed + i
            opts.require_sample_point_is_contained = True
            
            t0 = time.perf_counter()
            try:
                region = IrisNp(self.plant, self.plant_context, opts)
                self.regions.append(region)
            except Exception:
                pass
            self.build_time += time.perf_counter() - t0
        
        print(f"  Built {len(self.regions)} regions in {self.build_time:.1f}s", flush=True)
        return self.build_time
