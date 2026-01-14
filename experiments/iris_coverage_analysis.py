import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime
from enum import Enum

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    HPolyhedron,
    IrisOptions,
    Parser,
    IrisNp,
    RigidTransform,
    RollPitchYaw,
    SpatialInertia,
    UnitInertia,
    CoulombFriction,
    SaveIrisRegionsYamlFile
)
from pydrake.geometry import Box

from manipulation.scenarios import AddIiwa, AddWsg
from manipulation.utils import ConfigureParser


class SceneType(Enum):
    SINGLE_SHELF = 1
    TWO_SHELVES = 2
    TABLE_THREE_SHELVES = 3


@dataclass
class ExperimentConfig:
    seed_counts: List[int] = field(default_factory=lambda: [10, 20, 30, 50])
    num_trials: int = 3
    random_seed_base: int = 42
    iris_collision_samples: int = 3
    coverage_samples: int = 200  # Number of samples for coverage estimation
    output_dir: str = "iris_analysis_results_test_regions_saved"
    iris_regions_dir: str = "iris_regions"


@dataclass
class ExperimentResult:
    scene: str
    num_seeds: int
    trial: int
    num_collision_free: int
    num_regions_built: int
    num_isolated: int
    iris_time: float
    isolation_rate: float
    coverage_rate: float


def build_single_shelf():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
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


def build_two_shelves():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
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
    plant.Finalize()
    return builder.Build()


def add_table(plant, name, size_xy, top_thickness, top_z, rgba4):
    lx, ly = size_xy
    inertia = SpatialInertia(
        mass=30.0, p_PScm_E=[0, 0, 0],
        G_SP_E=UnitInertia.SolidBox(lx, ly, top_thickness),
    )
    body = plant.AddRigidBody(name, inertia)
    plant.WeldFrames(plant.world_frame(), body.body_frame(),
                     RigidTransform([0, 0, top_z - 0.5 * top_thickness]))
    friction = CoulombFriction(0.9, 0.8)
    plant.RegisterCollisionGeometry(body, RigidTransform(), Box(lx, ly, top_thickness),
                                    f"{name}_col", friction)
    plant.RegisterVisualGeometry(body, RigidTransform(), Box(lx, ly, top_thickness),
                                 f"{name}_vis", rgba4)


def build_table_three_shelves():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    
    z_table = 0.75
    add_table(plant, "table", (2.3, 1.8), 0.10, z_table, np.array([0.55, 0.25, 0.10, 1.0]))
    
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
    plant.Finalize()
    return builder.Build()


SCENE_BUILDERS = {
    SceneType.SINGLE_SHELF: build_single_shelf,
    SceneType.TWO_SHELVES: build_two_shelves,
    SceneType.TABLE_THREE_SHELVES: build_table_three_shelves,
}


def is_collision_free(diagram, context, plant, q) -> bool:
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q)
    sg = diagram.GetSubsystemByName("scene_graph")
    sg_context = sg.GetMyContextFromRoot(context)
    query = sg.get_query_output_port().Eval(sg_context)
    return not query.HasCollisions()


def sample_collision_free(diagram, context, plant, n, rng) -> List[np.ndarray]:
    q_lo = plant.GetPositionLowerLimits()
    q_hi = plant.GetPositionUpperLimits()
    configs = []
    for _ in range(n * 50):
        q = rng.uniform(q_lo, q_hi)
        if is_collision_free(diagram, context, plant, q):
            configs.append(q.copy())
            if len(configs) >= n:
                break
    return configs


def build_iris_regions(plant, plant_context, seeds, random_seed, collision_samples) -> Tuple[List[HPolyhedron], float]:
    regions = []
    total_time = 0.0
    
    for i, q in enumerate(seeds):
        plant.SetPositions(plant_context, q)
        opts = IrisOptions()
        opts.num_collision_infeasible_samples = collision_samples
        opts.random_seed = random_seed + i
        opts.require_sample_point_is_contained = True
        
        t0 = time.perf_counter()
        try:
            region = IrisNp(plant, plant_context, opts)
            regions.append(region)
        except Exception:
            pass
        total_time += time.perf_counter() - t0
    
    return regions, total_time


def count_isolated_regions(regions: List[HPolyhedron]) -> int:
    """Count regions that don't intersect with any other region."""
    n = len(regions)
    if n == 0:
        return 0
    
    isolated = 0
    for i in range(n):
        has_neighbor = False
        for j in range(n):
            if i != j and regions[i].IntersectsWith(regions[j]):
                has_neighbor = True
                break
        if not has_neighbor:
            isolated += 1
    return isolated


def estimate_coverage(regions: List[HPolyhedron], test_points: List[np.ndarray]) -> float:
    """Estimate coverage: fraction of test points inside at least one region."""
    if not regions or not test_points:
        return 0.0
    
    covered = 0
    for q in test_points:
        for region in regions:
            if region.PointInSet(q):
                covered += 1
                break
    return covered / len(test_points)


def run_experiment(scene_type: SceneType, num_seeds: int, trial: int, config: ExperimentConfig) -> ExperimentResult:
    rng = np.random.default_rng(config.random_seed_base + trial * 1000 + num_seeds)
    
    print(f"  Building scene...", flush=True)
    diagram = SCENE_BUILDERS[scene_type]()
    context = diagram.CreateDefaultContext()
    plant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)
    
    print(f"  Sampling {num_seeds} seed configs...", flush=True)
    seeds = sample_collision_free(diagram, context, plant, num_seeds, rng)
    print(f"  Found {len(seeds)} collision-free configs", flush=True)
    
    print(f"  Building IRIS regions...", flush=True)
    regions, iris_time = build_iris_regions(
        plant, plant_context, seeds, config.random_seed_base + trial, config.iris_collision_samples
    )
    print(f"  Built {len(regions)} regions in {iris_time:.1f}s", flush=True)
    
    print(f"  Checking intersections...", flush=True)
    isolated = count_isolated_regions(regions)
    isolation_rate = isolated / len(regions) if regions else 1.0
    print(f"  Isolated: {isolated}/{len(regions)} ({isolation_rate*100:.1f}%)", flush=True)
    
    # Estimate coverage with separate test points
    print(f"  Estimating coverage ({config.coverage_samples} samples)...", flush=True)
    rng_coverage = np.random.default_rng(config.random_seed_base + trial * 2000)
    test_points = sample_collision_free(diagram, context, plant, config.coverage_samples, rng_coverage)
    coverage_rate = estimate_coverage(regions, test_points)
    print(f"  Coverage: {coverage_rate*100:.1f}% ({len(test_points)} test points)", flush=True)

    save_iris_regions(regions, config.iris_regions_dir, scene_type, num_seeds, trial)
    
    return ExperimentResult(
        scene=scene_type.name,
        num_seeds=num_seeds,
        trial=trial,
        num_collision_free=len(seeds),
        num_regions_built=len(regions),
        num_isolated=isolated,
        iris_time=iris_time,
        isolation_rate=isolation_rate,
        coverage_rate=coverage_rate,
    )


def run_all_experiments(config: ExperimentConfig) -> List[ExperimentResult]:
    results = []
    scenes = [SceneType.SINGLE_SHELF, SceneType.TWO_SHELVES, SceneType.TABLE_THREE_SHELVES]
    total = len(scenes) * len(config.seed_counts) * config.num_trials
    current = 0

    os.makedirs(config.iris_regions_dir, exist_ok=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"IRIS Coverage Analysis: {total} experiments", flush=True)
    print(f"Seed counts: {config.seed_counts}", flush=True)
    print(f"Trials: {config.num_trials}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    for scene in scenes:
        for num_seeds in config.seed_counts:
            for trial in range(config.num_trials):
                current += 1
                print(f"[{current}/{total}] {scene.name} | seeds={num_seeds} | trial={trial+1}", flush=True)
                result = run_experiment(scene, num_seeds, trial, config)
                results.append(result)
                print("", flush=True)
    
    return results


def generate_statistics(results: List[ExperimentResult]) -> pd.DataFrame:
    df = pd.DataFrame([vars(r) for r in results])
    
    stats = df.groupby(['scene', 'num_seeds']).agg({
        'num_collision_free': 'mean',
        'num_regions_built': 'mean',
        'num_isolated': ['mean', 'std'],
        'isolation_rate': ['mean', 'std'],
        'coverage_rate': ['mean', 'std'],
        'iris_time': ['mean', 'std'],
    }).reset_index()
    
    stats.columns = ['scene', 'num_seeds', 'avg_collision_free', 'avg_regions',
                     'isolated_mean', 'isolated_std', 'isolation_rate_mean', 'isolation_rate_std',
                     'coverage_mean', 'coverage_std',
                     'iris_time_mean', 'iris_time_std']
    return stats


def print_results(stats: pd.DataFrame):
    print("\n" + "="*100, flush=True)
    print("RESULTS: IRIS Coverage Analysis", flush=True)
    print("="*100, flush=True)
    
    display = pd.DataFrame({
        'Scene': stats['scene'],
        'Seeds': stats['num_seeds'],
        'Regions': stats['avg_regions'].apply(lambda x: f"{x:.1f}"),
        'Isolated': stats.apply(lambda r: f"{r['isolated_mean']:.1f}±{r['isolated_std']:.1f}", axis=1),
        'Isolation%': stats.apply(lambda r: f"{r['isolation_rate_mean']*100:.0f}%", axis=1),
        'Coverage%': stats.apply(lambda r: f"{r['coverage_mean']*100:.1f}%±{r['coverage_std']*100:.1f}%", axis=1),
        'IRIS Time': stats.apply(lambda r: f"{r['iris_time_mean']:.0f}s", axis=1),
    })
    print(display.to_string(index=False), flush=True)
    print("="*100 + "\n", flush=True)

def save_iris_regions(regions: List[HPolyhedron], iris_regions_dir: str, scene_type: SceneType, num_seeds: int, trial: int):
    filename = f"{scene_type.name}_{num_seeds}_{trial + 1}.yaml"
    filepath = os.path.join(iris_regions_dir, filename)
    regions_dict = {f"region_{i}": region for i, region in enumerate(regions)}
    SaveIrisRegionsYamlFile(filepath, regions_dict)

def get_iris_regions_from_file(filepath: str) -> List[HPolyhedron]:
    regions_dict = LoadIrisRegionsYamlFile(filepath)
    return [regions_dict[f"region_{i}"] for i in range(len(regions_dict))]

def save_results(results: List[ExperimentResult], stats: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    raw_df = pd.DataFrame([vars(r) for r in results])
    raw_df.to_csv(os.path.join(output_dir, f"raw_{timestamp}.csv"), index=False)
    stats.to_csv(os.path.join(output_dir, f"stats_{timestamp}.csv"), index=False)
    print(f"Results saved to {output_dir}/", flush=True)


def main():
    import argparse
    import logging
    logging.getLogger("drake").setLevel(logging.WARNING)
    
    parser = argparse.ArgumentParser(description="IRIS coverage analysis")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30, 50])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=str, default="iris_analysis_results")
    args = parser.parse_args()
    
    config = ExperimentConfig(
        seed_counts=args.seeds,
        num_trials=args.trials,
        output_dir=args.output,
    )
    
    results = run_all_experiments(config)
    stats = generate_statistics(results)
    print_results(stats)
    save_results(results, stats, config.output_dir)


if __name__ == "__main__":
    main()
