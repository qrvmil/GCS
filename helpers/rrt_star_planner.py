"""
RRT* Path Planner for robot manipulator planning.

Implements RRT* algorithm with:
- Drake-based collision checking
- Configurable parameters
- Keypoint extraction from paths
"""

import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from experiments.scene_types import SceneType
from helpers.scene_builder import SceneBuilder


@dataclass
class RRTNode:
    config: np.ndarray
    parent_idx: Optional[int] = None
    cost: float = 0.0


class RRTStarPlanner:
    def __init__(self, scene_type: SceneType, 
                 max_iterations: int = 10000,
                 step_size: float = 0.5,
                 goal_sample_rate: float = 0.10,
                 neighbor_radius: float = 1.5,
                 random_seed: int = 42):
        self.scene_type = scene_type
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.neighbor_radius_base = neighbor_radius
        self.random_seed = random_seed
        
        self.diagram = SceneBuilder.build(scene_type)
        self.context = self.diagram.CreateDefaultContext()
        self.plant = self.diagram.GetSubsystemByName("plant")
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.scene_graph = self.diagram.GetSubsystemByName("scene_graph")
        
        self.q_lower = self.plant.GetPositionLowerLimits()
        self.q_upper = self.plant.GetPositionUpperLimits()
        self.nq = self.plant.num_positions()
        
        self._path: Optional[List[np.ndarray]] = None
        self._iterations_used: int = 0
        self._path_length: float = 0.0
        self._solve_time: float = 0.0
    
    def is_collision_free(self, q: np.ndarray) -> bool:
        self.plant.SetPositions(self.plant_context, q)
        sg_context = self.scene_graph.GetMyContextFromRoot(self.context)
        query = self.scene_graph.get_query_output_port().Eval(sg_context)
        return not query.HasCollisions()
    
    def is_edge_collision_free(self, q1: np.ndarray, q2: np.ndarray, 
                                num_checks: int = 15) -> bool:
        for t in np.linspace(0, 1, num_checks):
            q = q1 + t * (q2 - q1)
            if not self.is_collision_free(q):
                return False
        return True
    
    def _try_connect_to_goal(self, nodes: List[RRTNode], q_goal: np.ndarray,
                              goal_tolerance: float) -> Optional[int]:
        best_idx = None
        best_cost = float('inf')
        
        for i, node in enumerate(nodes):
            dist = np.linalg.norm(node.config - q_goal)
            if dist <= goal_tolerance * 3:
                if self.is_edge_collision_free(node.config, q_goal):
                    cost = node.cost + dist
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = i
        
        return best_idx
    
    def _sample_random(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.q_lower, self.q_upper)
    
    def _nearest_node_idx(self, nodes: List[RRTNode], q: np.ndarray) -> int:
        distances = [np.linalg.norm(node.config - q) for node in nodes]
        return int(np.argmin(distances))
    
    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        direction = q_to - q_from
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return q_to.copy()
        return q_from + (direction / dist) * self.step_size
    
    def _get_neighbor_radius(self, num_nodes: int) -> float:
        if num_nodes <= 1:
            return self.neighbor_radius_base
        gamma = self.neighbor_radius_base * 2.0
        radius = gamma * (np.log(num_nodes) / num_nodes) ** (1.0 / self.nq)
        return min(max(radius, self.step_size), self.neighbor_radius_base * 2)
    
    def _find_near_nodes(self, nodes: List[RRTNode], q: np.ndarray, 
                         radius: float) -> List[int]:
        near_indices = []
        for i, node in enumerate(nodes):
            if np.linalg.norm(node.config - q) <= radius:
                near_indices.append(i)
        return near_indices
    
    def _compute_cost(self, nodes: List[RRTNode], parent_idx: int, 
                      q_new: np.ndarray) -> float:
        return nodes[parent_idx].cost + np.linalg.norm(q_new - nodes[parent_idx].config)
    
    def _extract_path(self, nodes: List[RRTNode], goal_idx: int) -> List[np.ndarray]:
        path = []
        idx = goal_idx
        while idx is not None:
            path.append(nodes[idx].config.copy())
            idx = nodes[idx].parent_idx
        return list(reversed(path))
    
    def _compute_path_length(self, path: List[np.ndarray]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            total += np.linalg.norm(path[i+1] - path[i])
        return total
    
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray,
             goal_tolerance: float = 0.15) -> Optional[List[np.ndarray]]:
        rng = np.random.default_rng(self.random_seed)
        t0 = time.perf_counter()
        
        if not self.is_collision_free(q_start):
            print("---[RRT*] Start configuration is in collision")
            self._solve_time = time.perf_counter() - t0
            return None
        
        if not self.is_collision_free(q_goal):
            print("---[RRT*] Goal configuration is in collision")
            self._solve_time = time.perf_counter() - t0
            return None
        
        if self.is_edge_collision_free(q_start, q_goal, num_checks=20):
            print("---[RRT*] Direct path found!", flush=True)
            self._path = [q_start.copy(), q_goal.copy()]
            self._path_length = np.linalg.norm(q_goal - q_start)
            self._iterations_used = 0
            self._solve_time = time.perf_counter() - t0
            return self._path
        
        nodes: List[RRTNode] = [RRTNode(config=q_start.copy(), parent_idx=None, cost=0.0)]
        goal_idx: Optional[int] = None
        best_goal_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            if rng.random() < self.goal_sample_rate:
                q_rand = q_goal.copy()
            else:
                q_rand = self._sample_random(rng)
            
            nearest_idx = self._nearest_node_idx(nodes, q_rand)
            q_nearest = nodes[nearest_idx].config
            q_new = self._steer(q_nearest, q_rand)
            
            if not self.is_collision_free(q_new):
                continue
            
            radius = self._get_neighbor_radius(len(nodes))
            near_indices = self._find_near_nodes(nodes, q_new, radius)
            
            best_parent_idx = nearest_idx
            best_cost = self._compute_cost(nodes, nearest_idx, q_new)
            
            for near_idx in near_indices:
                if self.is_edge_collision_free(nodes[near_idx].config, q_new):
                    cost = self._compute_cost(nodes, near_idx, q_new)
                    if cost < best_cost:
                        best_cost = cost
                        best_parent_idx = near_idx
            
            if not self.is_edge_collision_free(nodes[best_parent_idx].config, q_new):
                continue
            
            new_node = RRTNode(config=q_new.copy(), parent_idx=best_parent_idx, cost=best_cost)
            new_idx = len(nodes)
            nodes.append(new_node)
            
            for near_idx in near_indices:
                if near_idx == best_parent_idx:
                    continue
                new_cost = best_cost + np.linalg.norm(q_new - nodes[near_idx].config)
                if new_cost < nodes[near_idx].cost:
                    if self.is_edge_collision_free(q_new, nodes[near_idx].config):
                        nodes[near_idx].parent_idx = new_idx
                        nodes[near_idx].cost = new_cost
            
            dist_to_goal = np.linalg.norm(q_new - q_goal)
            if dist_to_goal <= goal_tolerance:
                if self.is_edge_collision_free(q_new, q_goal):
                    goal_cost = best_cost + dist_to_goal
                    if goal_cost < best_goal_cost:
                        if goal_idx is None:
                            goal_node = RRTNode(config=q_goal.copy(), 
                                               parent_idx=new_idx, 
                                               cost=goal_cost)
                            goal_idx = len(nodes)
                            nodes.append(goal_node)
                        else:
                            nodes[goal_idx].parent_idx = new_idx
                            nodes[goal_idx].cost = goal_cost
                        best_goal_cost = goal_cost
            
            if goal_idx is None and (iteration + 1) % 500 == 0:
                connect_idx = self._try_connect_to_goal(nodes, q_goal, goal_tolerance)
                if connect_idx is not None:
                    dist = np.linalg.norm(nodes[connect_idx].config - q_goal)
                    goal_cost = nodes[connect_idx].cost + dist
                    goal_node = RRTNode(config=q_goal.copy(), 
                                       parent_idx=connect_idx, 
                                       cost=goal_cost)
                    goal_idx = len(nodes)
                    nodes.append(goal_node)
                    best_goal_cost = goal_cost
            
            if (iteration + 1) % 2000 == 0:
                status = f"found (cost={best_goal_cost:.2f})" if goal_idx else "searching"
                print(f"---[RRT*] Iteration {iteration + 1}/{self.max_iterations}, "
                      f"nodes={len(nodes)}, status={status}", flush=True)
        
        self._iterations_used = self.max_iterations
        self._solve_time = time.perf_counter() - t0
        
        if goal_idx is not None:
            self._path = self._extract_path(nodes, goal_idx)
            self._path_length = self._compute_path_length(self._path)
            return self._path
        
        return None
    
    def plan_bidirectional(self, q_start: np.ndarray, q_goal: np.ndarray,
                           goal_tolerance: float = 0.15) -> Optional[List[np.ndarray]]:
        rng = np.random.default_rng(self.random_seed)
        t0 = time.perf_counter()
        
        if not self.is_collision_free(q_start):
            print("  [BiRRT] Start configuration is in collision!")
            self._solve_time = time.perf_counter() - t0
            return None
        
        if not self.is_collision_free(q_goal):
            print("  [BiRRT] Goal configuration is in collision!")
            self._solve_time = time.perf_counter() - t0
            return None
        
        if self.is_edge_collision_free(q_start, q_goal, num_checks=20):
            print("  [BiRRT] Direct path found!", flush=True)
            self._path = [q_start.copy(), q_goal.copy()]
            self._path_length = np.linalg.norm(q_goal - q_start)
            self._iterations_used = 0
            self._solve_time = time.perf_counter() - t0
            return self._path
        
        tree_a: List[RRTNode] = [RRTNode(config=q_start.copy(), parent_idx=None, cost=0.0)]
        tree_b: List[RRTNode] = [RRTNode(config=q_goal.copy(), parent_idx=None, cost=0.0)]
        
        connection = None
        
        for iteration in range(self.max_iterations):
            q_rand = self._sample_random(rng)
            
            nearest_a = self._nearest_node_idx(tree_a, q_rand)
            q_new_a = self._steer(tree_a[nearest_a].config, q_rand)
            
            if self.is_collision_free(q_new_a) and \
               self.is_edge_collision_free(tree_a[nearest_a].config, q_new_a):
                cost_a = tree_a[nearest_a].cost + np.linalg.norm(q_new_a - tree_a[nearest_a].config)
                tree_a.append(RRTNode(config=q_new_a.copy(), parent_idx=nearest_a, cost=cost_a))
                idx_a = len(tree_a) - 1
                
                nearest_b = self._nearest_node_idx(tree_b, q_new_a)
                dist = np.linalg.norm(tree_b[nearest_b].config - q_new_a)
                
                if dist <= goal_tolerance and \
                   self.is_edge_collision_free(tree_b[nearest_b].config, q_new_a):
                    connection = (idx_a, nearest_b)
                    break
                
                q_new_b = self._steer(tree_b[nearest_b].config, q_new_a)
                if self.is_collision_free(q_new_b) and \
                   self.is_edge_collision_free(tree_b[nearest_b].config, q_new_b):
                    cost_b = tree_b[nearest_b].cost + np.linalg.norm(q_new_b - tree_b[nearest_b].config)
                    tree_b.append(RRTNode(config=q_new_b.copy(), parent_idx=nearest_b, cost=cost_b))
                    idx_b = len(tree_b) - 1
                    
                    dist = np.linalg.norm(q_new_b - q_new_a)
                    if dist <= goal_tolerance and \
                       self.is_edge_collision_free(q_new_a, q_new_b):
                        connection = (idx_a, idx_b)
                        break
            
            tree_a, tree_b = tree_b, tree_a
            
            if (iteration + 1) % 2000 == 0:
                print(f"  [BiRRT] Iteration {iteration + 1}/{self.max_iterations}, "
                      f"tree_a={len(tree_a)}, tree_b={len(tree_b)}", flush=True)
        
        self._iterations_used = iteration + 1
        self._solve_time = time.perf_counter() - t0
        
        if connection is None:
            return None
        
        idx_a, idx_b = connection
        
        path_a = []
        idx = idx_a
        while idx is not None:
            path_a.append(tree_a[idx].config.copy())
            idx = tree_a[idx].parent_idx
        path_a = list(reversed(path_a))
        
        path_b = []
        idx = idx_b
        while idx is not None:
            path_b.append(tree_b[idx].config.copy())
            idx = tree_b[idx].parent_idx
        
        self._path = path_a + path_b
        
        start_dist = np.linalg.norm(self._path[0] - q_start)
        if start_dist > np.linalg.norm(self._path[-1] - q_start):
            self._path = list(reversed(self._path))
        
        self._path_length = self._compute_path_length(self._path)
        return self._path
    
    @property
    def path(self) -> Optional[List[np.ndarray]]:
        return self._path
    
    @property
    def iterations_used(self) -> int:
        return self._iterations_used
    
    @property
    def path_length(self) -> float:
        return self._path_length
    
    @property
    def solve_time(self) -> float:
        return self._solve_time


def extract_keypoints(path: List[np.ndarray], 
                      max_points: int = 10,
                      epsilon: float = 0.15,
                      min_distance: float = 0.1) -> List[np.ndarray]:
    if len(path) <= 2:
        return [p.copy() for p in path]
    
    simplified = _douglas_peucker(path, epsilon)
    
    if len(simplified) > max_points:
        simplified = _reduce_to_max_points(simplified, max_points)
    
    simplified = _enforce_min_distance(simplified, min_distance)
    
    if len(simplified) < 2:
        simplified = [path[0].copy(), path[-1].copy()]
    
    return simplified


def _douglas_peucker(path: List[np.ndarray], epsilon: float) -> List[np.ndarray]:
    if len(path) <= 2:
        return [p.copy() for p in path]
    
    start = path[0]
    end = path[-1]
    
    max_dist = 0.0
    max_idx = 0
    
    for i in range(1, len(path) - 1):
        dist = _point_line_distance(path[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    if max_dist > epsilon:
        left = _douglas_peucker(path[:max_idx + 1], epsilon)
        right = _douglas_peucker(path[max_idx:], epsilon)
        
        return left[:-1] + right
    else:
        return [start.copy(), end.copy()]


def _point_line_distance(point: np.ndarray, line_start: np.ndarray, 
                         line_end: np.ndarray) -> float:
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        return np.linalg.norm(point - line_start)
    
    t = np.clip(np.dot(point - line_start, line_vec) / (line_len ** 2), 0, 1)
    projection = line_start + t * line_vec
    
    return np.linalg.norm(point - projection)


def _reduce_to_max_points(path: List[np.ndarray], max_points: int) -> List[np.ndarray]:
    if len(path) <= max_points:
        return path
    
    n = len(path)
    
    curvatures = np.zeros(n)
    for i in range(1, n - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 1e-10 and norm2 > 1e-10:
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
            curvatures[i] = 1.0 - cos_angle
    
    selected_indices = {0, n - 1}
    
    remaining = max_points - 2
    if remaining > 0:
        interior_indices = list(range(1, n - 1))
        interior_indices.sort(key=lambda i: curvatures[i], reverse=True)
        
        for idx in interior_indices[:remaining]:
            selected_indices.add(idx)
    
    selected_list = sorted(selected_indices)
    return [path[i].copy() for i in selected_list]


def _enforce_min_distance(path: List[np.ndarray], 
                          min_distance: float) -> List[np.ndarray]:
    if len(path) <= 2:
        return path
    
    result = [path[0].copy()]
    
    for i in range(1, len(path) - 1):
        if np.linalg.norm(path[i] - result[-1]) >= min_distance:
            result.append(path[i].copy())
    
    result.append(path[-1].copy())
    
    return result


def extract_keypoints_uniform(path: List[np.ndarray], 
                              num_points: int = 10) -> List[np.ndarray]:
    if len(path) <= num_points:
        return [p.copy() for p in path]
    
    cum_distances = [0.0]
    for i in range(1, len(path)):
        cum_distances.append(cum_distances[-1] + np.linalg.norm(path[i] - path[i-1]))
    
    total_length = cum_distances[-1]
    if total_length < 1e-10:
        return [path[0].copy(), path[-1].copy()]
    
    keypoints = [path[0].copy()]
    
    for i in range(1, num_points - 1):
        target_dist = (i / (num_points - 1)) * total_length
        
        for j in range(1, len(path)):
            if cum_distances[j] >= target_dist:
                t = (target_dist - cum_distances[j-1]) / (cum_distances[j] - cum_distances[j-1])
                q = path[j-1] + t * (path[j] - path[j-1])
                keypoints.append(q)
                break
    
    keypoints.append(path[-1].copy())
    return keypoints
