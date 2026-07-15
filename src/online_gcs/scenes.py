from enum import Enum
from importlib.resources import files

import numpy as np
import yaml
from manipulation.scenarios import AddIiwa, AddWsg
from manipulation.utils import ConfigureParser
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    DiagramBuilder,
    Parser,
    RigidTransform,
    RollPitchYaw,
    SpatialInertia,
    UnitInertia,
)
from pydrake.geometry import Box


class SceneType(Enum):
    SINGLE_SHELF = 1
    TWO_SHELVES = 2
    TABLE_THREE_SHELVES = 3


_CONFIG_KEYS = {
    SceneType.SINGLE_SHELF: "positions_single_shelf_cspace",
    SceneType.TWO_SHELVES: "positions_two_shelves_cspace",
    SceneType.TABLE_THREE_SHELVES: "positions_three_shelves_cspace",
}


def load_shelf_configurations(scene_type: SceneType) -> list[np.ndarray]:
    resource = files("online_gcs.resources").joinpath("shelf_cspace_positions.yaml")
    with resource.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return [np.asarray(row, dtype=float) for row in payload[_CONFIG_KEYS[scene_type]]]


class SceneBuilder:
    @staticmethod
    def build(scene_type: SceneType):
        builders = {
            SceneType.SINGLE_SHELF: SceneBuilder._single_shelf,
            SceneType.TWO_SHELVES: SceneBuilder._two_shelves,
            SceneType.TABLE_THREE_SHELVES: SceneBuilder._table_three_shelves,
        }
        return builders[scene_type]()

    @staticmethod
    def _single_shelf():
        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
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
        plant.Finalize()
        return builder.Build()

    @staticmethod
    def _two_shelves():
        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        iiwa = AddIiwa(plant)
        AddWsg(plant, iiwa, welded=True, sphere=False)
        parser = Parser(plant)
        ConfigureParser(parser)

        s1 = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
        plant.RenameModelInstance(s1, "shelves1")
        s2 = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
        plant.RenameModelInstance(s2, "shelves2")

        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("shelves_body", s1),
            RigidTransform([0.95, -0.35, 0.40]),
        )
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("shelves_body", s2),
            RigidTransform([0.95, 0.35, 0.40]),
        )
        plant.Finalize()
        return builder.Build()

    @staticmethod
    def _table_three_shelves():
        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

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

        z_sh = z_table + 0.40
        for name, xyz, yaw in [
            ("sl", [0.85, 0.65, z_sh], 45),
            ("sr", [0.85, -0.65, z_sh], -45),
            ("sb", [-0.95, 0.0, z_sh], 180),
        ]:
            model = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
            plant.RenameModelInstance(model, name)
            plant.WeldFrames(
                plant.world_frame(),
                plant.GetBodyByName("shelves_body", model).body_frame(),
                RigidTransform(RollPitchYaw(0, 0, np.deg2rad(yaw)), xyz),
            )
        plant.Finalize()
        return builder.Build()
