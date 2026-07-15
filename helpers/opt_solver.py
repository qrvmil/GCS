import time
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from pydrake.all import (
    MathematicalProgram,
    Solve,
    PiecewisePolynomial,
    SnoptSolver,
)
from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint

from online_gcs.scenes import SceneBuilder, SceneType


@dataclass
class TrajOptResult:
    success: bool
    path: Optional[List[np.ndarray]] = None
    trajectory: Optional[PiecewisePolynomial] = None
    path_length: float = 0.0
    initial_path_length: float = 0.0
    solve_time: float = 0.0
    min_distance: float = 0.0
    solver_info: str = ""


class TrajOptSolver:
    def __init__(
        self,
        scene_type: SceneType,
        num_knots: int = 21,
        d_min: float = 0.01,
        smoothness_weight: float = 1.0,
        path_length_weight: float = 0.1,
        influence_distance: float = 0.3,
        major_iter_limit: int = 3000,
        feasibility_tol: float = 1e-5,
        optimality_tol: float = 1e-3,
    ):
        self.scene_type = scene_type
        self.num_knots = num_knots
        self.d_min = d_min
        self.smoothness_weight = smoothness_weight
        self.path_length_weight = path_length_weight
        self.influence_distance = influence_distance
        self.major_iter_limit = major_iter_limit
        self.feasibility_tol = feasibility_tol
        self.optimality_tol = optimality_tol

        self.diagram = SceneBuilder.build(scene_type)
        self.context = self.diagram.CreateDefaultContext()
        self.plant = self.diagram.GetSubsystemByName("plant")
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.scene_graph = self.diagram.GetSubsystemByName("scene_graph")

        self.q_lower = self.plant.GetPositionLowerLimits()
        self.q_upper = self.plant.GetPositionUpperLimits()
        self.nq = self.plant.num_positions()

    def _resample_path(self, path: List[np.ndarray], N: int) -> np.ndarray:
        cum_dist = [0.0]
        for i in range(1, len(path)):
            cum_dist.append(cum_dist[-1] + np.linalg.norm(path[i] - path[i - 1]))
        total_length = cum_dist[-1]

        if total_length < 1e-10:
            return np.column_stack([path[0].copy() for _ in range(N)])

        resampled = []
        for k in range(N):
            target = (k / (N - 1)) * total_length
            for j in range(1, len(path)):
                if cum_dist[j] >= target:
                    t = (target - cum_dist[j - 1]) / (cum_dist[j] - cum_dist[j - 1])
                    resampled.append(path[j - 1] + t * (path[j] - path[j - 1]))
                    break
            else:
                resampled.append(path[-1].copy())

        return np.column_stack(resampled)

    @staticmethod
    def _path_length_from_matrix(Q: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(Q, axis=1), axis=0)))

    def _check_min_distance_along(self, Q: np.ndarray) -> float:
        min_d = float("inf")
        for k in range(Q.shape[1]):
            self.plant.SetPositions(self.plant_context, Q[:, k])
            sg_ctx = self.scene_graph.GetMyContextFromRoot(self.context)
            query = self.scene_graph.get_query_output_port().Eval(sg_ctx)
            for p in query.ComputeSignedDistancePairwiseClosestPoints():
                if p.distance < min_d:
                    min_d = p.distance
        return min_d

    def _densely_check_collision(self, Q: np.ndarray,
                                 subdivisions: int = 4) -> float:
        """Check min distance along the path including intermediate points
        between each pair of knots (linear interpolation)."""
        N = Q.shape[1]
        min_d = float("inf")
        for k in range(N - 1):
            for s in range(subdivisions + 1):
                alpha = s / subdivisions
                q = (1.0 - alpha) * Q[:, k] + alpha * Q[:, k + 1]
                self.plant.SetPositions(self.plant_context, q)
                sg_ctx = self.scene_graph.GetMyContextFromRoot(self.context)
                query = self.scene_graph.get_query_output_port().Eval(sg_ctx)
                for p in query.ComputeSignedDistancePairwiseClosestPoints():
                    if p.distance < min_d:
                        min_d = p.distance
        return min_d

    def optimize_rrt_path(self, rrt_path: List[np.ndarray]) -> TrajOptResult:
        t0 = time.perf_counter()

        N = self.num_knots
        nq = self.nq
        q_start = rrt_path[0].copy()
        q_goal = rrt_path[-1].copy()

        seed = self._resample_path(rrt_path, N)
        initial_path_length = self._path_length_from_matrix(seed)

        prog = MathematicalProgram()
        Q = prog.NewContinuousVariables(nq, N, "Q")

        for k in range(N):
            prog.SetInitialGuess(Q[:, k], seed[:, k])

        for i in range(nq):
            prog.AddLinearEqualityConstraint(Q[i, 0], q_start[i])
            prog.AddLinearEqualityConstraint(Q[i, N - 1], q_goal[i])

        for k in range(N):
            prog.AddBoundingBoxConstraint(self.q_lower, self.q_upper, Q[:, k])

        I_nq = np.eye(nq)
        A_smooth = np.hstack([I_nq, -2.0 * I_nq, I_nq])
        H_smooth = 2.0 * self.smoothness_weight * (A_smooth.T @ A_smooth)
        b_smooth = np.zeros(3 * nq)

        for k in range(1, N - 1):
            x_block = np.concatenate([Q[:, k - 1], Q[:, k], Q[:, k + 1]])
            prog.AddQuadraticCost(H_smooth, b_smooth, x_block)

        A_len = np.hstack([-I_nq, I_nq])
        H_len = 2.0 * self.path_length_weight * (A_len.T @ A_len)
        b_len = np.zeros(2 * nq)

        for k in range(N - 1):
            x_block = np.concatenate([Q[:, k], Q[:, k + 1]])
            prog.AddQuadraticCost(H_len, b_len, x_block)

        collision_constraint = MinimumDistanceLowerBoundConstraint(
            plant=self.plant,
            bound=self.d_min,
            plant_context=self.plant_context,
            influence_distance_offset=self.influence_distance,
        )

        # Collision constraints at every knot point
        for k in range(N):
            prog.AddConstraint(collision_constraint, Q[:, k])

        # Midpoint auxiliary variables — collision checking between knots
        M = prog.NewContinuousVariables(nq, N - 1, "M")
        Aeq_mid = np.hstack([I_nq, -0.5 * I_nq, -0.5 * I_nq])
        beq_mid = np.zeros(nq)

        for k in range(N - 1):
            prog.SetInitialGuess(M[:, k], 0.5 * (seed[:, k] + seed[:, k + 1]))
            prog.AddBoundingBoxConstraint(self.q_lower, self.q_upper, M[:, k])
            vars_mid = np.concatenate([M[:, k], Q[:, k], Q[:, k + 1]])
            prog.AddLinearEqualityConstraint(Aeq_mid, beq_mid, vars_mid)
            prog.AddConstraint(collision_constraint, M[:, k])

        snopt = SnoptSolver()
        snopt_id = snopt.id()
        prog.SetSolverOption(snopt_id, "Major iterations limit", self.major_iter_limit)
        prog.SetSolverOption(snopt_id, "Major feasibility tolerance", self.feasibility_tol)
        prog.SetSolverOption(snopt_id, "Major optimality tolerance", self.optimality_tol)

        result = snopt.Solve(prog)
        solve_time = time.perf_counter() - t0

        snopt_info = result.get_solver_details().info
        usable = result.is_success() or snopt_info in (1, 2)

        if usable:
            Q_opt = result.GetSolution(Q)
            opt_length = self._path_length_from_matrix(Q_opt)
            min_dist = self._densely_check_collision(Q_opt, subdivisions=4)

            if min_dist < 0:
                print(
                    f"[Opt solver] REJECTED (collision)  min_dist={min_dist:.4f}  "
                    f"time={solve_time:.3f}s  snopt_info={snopt_info}",
                    flush=True,
                )
                return TrajOptResult(
                    success=False,
                    solve_time=solve_time,
                    initial_path_length=initial_path_length,
                    min_distance=min_dist,
                    solver_info=f"Rejected: collision min_dist={min_dist:.4f}",
                )

            path_opt = [Q_opt[:, k].copy() for k in range(N)]
            times = np.linspace(0.0, 1.0, N)
            trajectory = PiecewisePolynomial.FirstOrderHold(times, Q_opt)

            print(
                f"[Opt solver] SUCCESS  length {initial_path_length:.3f} -> "
                f"{opt_length:.3f}  min_dist={min_dist:.4f}  "
                f"time={solve_time:.3f}s  snopt_info={snopt_info}",
                flush=True,
            )
            return TrajOptResult(
                success=True,
                path=path_opt,
                trajectory=trajectory,
                path_length=opt_length,
                initial_path_length=initial_path_length,
                solve_time=solve_time,
                min_distance=min_dist,
                solver_info=f"SNOPT(info={snopt_info})",
            )
        else:
            print(
                f"[Opt solver] FAILED  time={solve_time:.3f}s  "
                f"snopt_info={snopt_info}",
                flush=True,
            )
            return TrajOptResult(
                success=False,
                solve_time=solve_time,
                initial_path_length=initial_path_length,
                solver_info=f"Failed: SNOPT(info={snopt_info})",
            )
