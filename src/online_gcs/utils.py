from typing import List

import numpy as np
from pydrake.all import Context, MultibodyPlant
from pydrake.multibody.inverse_kinematics import (
    InverseKinematics,
    MinimumDistanceLowerBoundConstraint,
)
from pydrake.solvers import Solve


def solve_IK(
    plant: MultibodyPlant,
    plant_context: Context,
    goal_pos: List[float],
    ee_frame,
    q_initial: np.ndarray,
    min_clearance: float = 0.00,
) -> np.ndarray | None:
    """
    IK с учётом коллизий:
    - min_clearance >= 0: минимальная допустимая дистанция между всеми геометриями.
    - если решения нет → возвращает None.
    """
    plant.SetPositions(plant_context, q_initial)

    ik = InverseKinematics(plant, plant_context)
    q = ik.q()
    prog = ik.prog()
    distance_constraint = MinimumDistanceLowerBoundConstraint(
        plant=plant,
        bound=min_clearance,
        plant_context=plant_context,
        influence_distance_offset=0.01,
    )
    prog.AddConstraint(distance_constraint, q)

    p_BQ = np.array([0.0, 0.1, 0.0])
    p_WQ = np.array(goal_pos, dtype=float)

    ik.AddPositionConstraint(
        ee_frame,
        p_BQ,
        plant.world_frame(),
        p_WQ,
        p_WQ,
    )

    Q = np.eye(len(q_initial))
    prog.AddQuadraticErrorCost(Q, q_initial, q)
    prog.SetInitialGuess(q, q_initial)

    result = Solve(prog)
    if not result.is_success():
        return None

    q_sol = result.GetSolution(q)
    plant.SetPositions(plant_context, q_sol)
    return q_sol
