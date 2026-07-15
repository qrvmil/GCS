import time
import multiprocessing
import queue
from typing import Any, List

import numpy as np
from pydrake.all import HPolyhedron, IrisOptions, IrisNp

from online_gcs.scenes import SceneBuilder, SceneType


def _exploration_worker(
    scene_type_value: int,
    random_seed: int,
    task_queue: Any,
    result_queue: Any,
    collision_samples: int = 3,
) -> None:
    scene_type = SceneType(scene_type_value)
    diagram = SceneBuilder.build(scene_type)
    context = diagram.CreateDefaultContext()
    plant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    while True:
        try:
            task = task_queue.get()
        except Exception:
            continue
        if task is None:
            break

        task_id = task["task_id"]
        interval_id = task["interval_id"]
        seed_q = np.array(task["seed_q"], dtype=np.float64)

        build_time = 0.0
        success = False
        A, b = None, None

        try:
            plant.SetPositions(plant_context, seed_q)
            opts = IrisOptions()
            opts.num_collision_infeasible_samples = collision_samples
            opts.random_seed = random_seed + task_id
            opts.require_sample_point_is_contained = True

            t0 = time.perf_counter()
            region = IrisNp(plant, plant_context, opts)
            build_time = time.perf_counter() - t0
            A = np.array(region.A())
            b = np.array(region.b())
            success = True
        except Exception:
            pass

        result_queue.put(
            {
                "task_id": task_id,
                "interval_id": interval_id,
                "success": success,
                "seed_q": seed_q,
                "A": A,
                "b": b,
                "build_time": build_time,
            }
        )


class ParallelExplorationCoordinator:
    def __init__(
        self,
        scene_type: SceneType,
        num_workers: int = 2,
        random_seed: int = 42,
        collision_samples: int = 3,
    ):
        self.scene_type = scene_type
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.collision_samples = collision_samples
        self._ctx = multiprocessing.get_context("spawn")
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._processes: List[multiprocessing.Process] = []
        self._task_counter = 0

        for _ in range(num_workers):
            p = self._ctx.Process(
                target=_exploration_worker,
                args=(
                    scene_type.value,
                    random_seed,
                    self._task_queue,
                    self._result_queue,
                    collision_samples,
                ),
            )
            p.start()
            self._processes.append(p)

    def generate_exploration_seeds(
        self,
        existing_regions: List[HPolyhedron],
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        rng: np.random.Generator,
        count: int,
    ) -> List[np.ndarray]:
        seeds = []
        attempts = 0
        max_attempts = count * 50
        while len(seeds) < count and attempts < max_attempts:
            attempts += 1
            q = rng.uniform(q_lower, q_upper)
            if not existing_regions:
                seeds.append(q)
                continue
            inside_any = False
            for r in existing_regions:
                try:
                    if r.PointInSet(q):
                        inside_any = True
                        break
                except Exception:
                    pass
            if not inside_any:
                seeds.append(q)
        return seeds

    def submit_tasks(self, seeds: List[np.ndarray], interval_id: int = 0) -> None:
        for seed_q in seeds:
            self._task_counter += 1
            self._task_queue.put(
                {
                    "task_id": self._task_counter,
                    "interval_id": interval_id,
                    "seed_q": np.asarray(seed_q).tolist(),
                }
            )

    def _is_near_duplicate(
        self,
        new_region: HPolyhedron,
        existing_regions: List[HPolyhedron],
        rng: np.random.Generator,
    ) -> bool:
        if not existing_regions:
            return False
        for existing in existing_regions:
            try:
                center = new_region.ChebyshevCenter()
                if existing.PointInSet(center):
                    samples_inside = 0
                    total_samples = 10
                    for _ in range(total_samples):
                        sample = new_region.UniformSample(rng)
                        if existing.PointInSet(sample):
                            samples_inside += 1
                    if samples_inside >= total_samples * 0.8:
                        return True
            except Exception:
                pass
        return False

    def collect_results(
        self,
        existing_regions: List[HPolyhedron],
        rng: np.random.Generator,
    ) -> List[HPolyhedron]:
        accepted: List[HPolyhedron] = []
        try:
            while True:
                result = self._result_queue.get_nowait()
                if not result.get("success") or result.get("A") is None or result.get("b") is None:
                    continue
                seed_q = np.array(result["seed_q"])
                A = np.array(result["A"])
                b = np.array(result["b"])
                new_region = HPolyhedron(A, b)

                seed_inside = False
                for r in existing_regions:
                    try:
                        if r.PointInSet(seed_q):
                            seed_inside = True
                            break
                    except Exception:
                        pass
                if seed_inside:
                    continue

                all_regions = existing_regions + accepted
                if self._is_near_duplicate(new_region, all_regions, rng):
                    continue

                accepted.append(new_region)
        except queue.Empty:
            pass
        return accepted

    def shutdown(self, timeout: float = 5.0) -> None:
        for _ in self._processes:
            try:
                self._task_queue.put(None)
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        self._processes.clear()
