"""Meshcat visualization support for the online planner."""

import colorsys
import time
from collections.abc import Sequence

import numpy as np
from manipulation.scenarios import AddIiwa, AddWsg
from manipulation.utils import ConfigureParser
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    Cylinder,
    DiagramBuilder,
    Parser,
    Rgba,
    RigidTransform,
    RollPitchYaw,
    SpatialInertia,
    Sphere,
    StartMeshcat,
    UnitInertia,
)
from pydrake.visualization import AddDefaultVisualization

from online_gcs.scenes import SceneType


class MeshcatVisualizer:
    """Own the Meshcat scene and all rendering state for an online run."""

    def __init__(self, scene_type: SceneType, current_qpos: np.ndarray):
        print("Setting up Meshcat visualization...", flush=True)

        self.meshcat = StartMeshcat()

        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

        if scene_type == SceneType.SINGLE_SHELF:
            self._build_single_shelf_scene(plant)
        elif scene_type == SceneType.TWO_SHELVES:
            self._build_two_shelves_scene(plant)
        elif scene_type == SceneType.TABLE_THREE_SHELVES:
            self._build_table_three_shelves_scene(plant)

        plant.Finalize()
        AddDefaultVisualization(builder, self.meshcat)

        self._diagram = builder.Build()
        self._context = self._diagram.CreateDefaultContext()
        self._plant = self._diagram.GetSubsystemByName("plant")
        self._plant_context = self._plant.GetMyContextFromRoot(self._context)

        self._plant.SetPositions(self._plant_context, current_qpos)
        self._diagram.ForcedPublish(self._context)

        self.target_point_count = 0
        self.path_line_count = 0

        print(f"Meshcat URL: {self.meshcat.web_url()}", flush=True)

    @staticmethod
    def _build_single_shelf_scene(plant) -> None:
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

    @staticmethod
    def _build_two_shelves_scene(plant) -> None:
        iiwa = AddIiwa(plant)
        AddWsg(plant, iiwa, welded=True, sphere=False)
        parser = Parser(plant)
        ConfigureParser(parser)

        shelf_one = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
        plant.RenameModelInstance(shelf_one, "shelves1")
        shelf_two = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
        plant.RenameModelInstance(shelf_two, "shelves2")

        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("shelves_body", shelf_one),
            RigidTransform([0.95, -0.35, 0.40]),
        )
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("shelves_body", shelf_two),
            RigidTransform([0.95, 0.35, 0.40]),
        )

    @staticmethod
    def _build_table_three_shelves_scene(plant) -> None:
        z_table = 0.75
        lx, ly, thickness = 2.3, 1.8, 0.10
        inertia = SpatialInertia(
            mass=30.0,
            p_PScm_E=[0, 0, 0],
            G_SP_E=UnitInertia.SolidBox(lx, ly, thickness),
        )
        table = plant.AddRigidBody("table", inertia)
        plant.WeldFrames(
            plant.world_frame(),
            table.body_frame(),
            RigidTransform([0, 0, z_table - 0.5 * thickness]),
        )
        friction = CoulombFriction(0.9, 0.8)
        plant.RegisterCollisionGeometry(
            table,
            RigidTransform(),
            Box(lx, ly, thickness),
            "table_col",
            friction,
        )
        plant.RegisterVisualGeometry(
            table,
            RigidTransform(),
            Box(lx, ly, thickness),
            "table_vis",
            np.array([0.55, 0.25, 0.10, 1.0]),
        )

        parser = Parser(plant)
        ConfigureParser(parser)

        iiwa = parser.AddModelsFromUrl(
            "package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf"
        )[0]
        plant.RenameModelInstance(iiwa, "iiwa")
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetBodyByName("iiwa_link_0", iiwa).body_frame(),
            RigidTransform([0, 0, z_table]),
        )
        AddWsg(plant, iiwa, welded=True, sphere=False)

        z_shelf = z_table + 0.40
        for name, xyz, yaw in [
            ("sl", [0.85, 0.65, z_shelf], 45),
            ("sr", [0.85, -0.65, z_shelf], -45),
            ("sb", [-0.95, 0.0, z_shelf], 180),
        ]:
            model = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
            plant.RenameModelInstance(model, name)
            plant.WeldFrames(
                plant.world_frame(),
                plant.GetBodyByName("shelves_body", model).body_frame(),
                RigidTransform(RollPitchYaw(0, 0, np.deg2rad(yaw)), xyz),
            )

    def web_url(self) -> str:
        """Return the URL of this visualizer's Meshcat instance."""
        return self.meshcat.web_url()

    def update_robot(self, q: np.ndarray) -> None:
        """Update the rendered robot configuration."""
        self._plant.SetPositions(self._plant_context, q)
        self._diagram.ForcedPublish(self._context)

    def _get_ee_position(self, q: np.ndarray) -> np.ndarray:
        self._plant.SetPositions(self._plant_context, q)
        try:
            wsg = self._plant.GetModelInstanceByName("gripper")
            gripper_frame = self._plant.GetFrameByName("body", wsg)
            return self._plant.CalcPointsPositions(
                self._plant_context,
                gripper_frame,
                np.array([[0], [0.1], [0]]),
                self._plant.world_frame(),
            ).flatten()
        except Exception:
            return np.zeros(3)

    def draw_target(self, q_target: np.ndarray, color: Rgba | None = None) -> None:
        """Draw a target configuration at its projected end-effector position."""
        ee_pos = self._get_ee_position(q_target)

        if color is None:
            hue = (self.target_point_count * 0.618033988749895) % 1.0
            color = self._hsv_to_rgba(hue, 0.8, 0.9, 0.8)

        point_name = f"targets/target_{self.target_point_count}"
        self.meshcat.SetObject(point_name, Sphere(0.025), color)
        self.meshcat.SetTransform(point_name, RigidTransform(ee_pos))
        self.target_point_count += 1

    def draw_path(
        self,
        path: Sequence[np.ndarray],
        color: Rgba | None = None,
        use_gcs: bool = True,
    ) -> None:
        """Draw a configuration-space path using projected end-effector segments."""
        if len(path) < 2:
            return

        if color is None:
            color = Rgba(0.2, 0.8, 0.2, 0.7) if use_gcs else Rgba(0.9, 0.5, 0.1, 0.7)

        for i in range(len(path) - 1):
            pos1 = self._get_ee_position(path[i])
            pos2 = self._get_ee_position(path[i + 1])
            self._draw_line_segment(
                pos1,
                pos2,
                color,
                f"paths/path_{self.path_line_count}/seg_{i}",
            )

        self.path_line_count += 1

    def _draw_line_segment(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        color: Rgba,
        name: str,
    ) -> None:
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
            rotation = RollPitchYaw(
                0,
                np.arctan2(
                    np.sqrt(direction_normalized[0] ** 2 + direction_normalized[1] ** 2),
                    direction_normalized[2],
                ),
                np.arctan2(direction_normalized[1], direction_normalized[0]),
            )

        self.meshcat.SetObject(name, Cylinder(0.003, length), color)
        self.meshcat.SetTransform(name, RigidTransform(rotation, midpoint))

    def animate_trajectory(
        self,
        trajectory,
        duration: float = 2.0,
        num_samples: int = 50,
    ) -> None:
        """Animate the robot along a trajectory."""
        if trajectory is None:
            return

        times = np.linspace(trajectory.start_time(), trajectory.end_time(), num_samples)
        for sample_time in times:
            q = trajectory.value(sample_time).flatten()
            self.update_robot(q)
            time.sleep(duration / num_samples)

    def animate_path(self, path: Sequence[np.ndarray], duration: float = 2.0) -> None:
        """Animate the robot along a list of configurations."""
        if not path:
            return

        delay = duration / len(path)
        for q in path:
            self.update_robot(q)
            time.sleep(delay)

    @staticmethod
    def _hsv_to_rgba(h: float, s: float, v: float, a: float = 1.0) -> Rgba:
        red, green, blue = colorsys.hsv_to_rgb(h, s, v)
        return Rgba(red, green, blue, a)

    def clear_paths(self) -> None:
        """Clear all rendered paths."""
        self.meshcat.Delete("paths")
        self.path_line_count = 0

    def clear_figure_artifacts(self) -> None:
        """Clear paper-figure overlays while preserving the scene itself."""
        self.meshcat.Delete("targets")
        self.meshcat.Delete("paths")
        self.meshcat.Delete("figure")
        self.target_point_count = 0
        self.path_line_count = 0

    def draw_config_marker(
        self,
        q: np.ndarray,
        color: Rgba,
        name: str,
        radius: float = 0.02,
    ) -> None:
        """Draw a sphere at a configuration's projected end-effector position."""
        ee_pos = self._get_ee_position(q)
        self.meshcat.SetObject(name, Sphere(radius), color)
        self.meshcat.SetTransform(name, RigidTransform(ee_pos))

    def draw_config_markers(
        self,
        configs: Sequence[np.ndarray],
        group_name: str,
        color: Rgba,
        radius: float = 0.018,
    ) -> None:
        """Draw projected end-effector markers for multiple configurations."""
        for i, q in enumerate(configs):
            self.draw_config_marker(
                q=q,
                color=color,
                name=f"{group_name}/cfg_{i}",
                radius=radius,
            )
