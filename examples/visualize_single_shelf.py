import time

from online_gcs import OnlineGCS, SceneType


def main() -> None:
    planner = OnlineGCS(
        scene_type=SceneType.SINGLE_SHELF,
        random_seed=42,
        max_iterations=1,
        visualize=True,
    )
    summary = planner.run(num_keypoints=2, prune_interval=0)
    print(
        {key: summary[key] for key in ("total_queries", "gcs_success_count", "rrt_fallback_count")}
    )
    print("Meshcat remains available; press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nVisualization stopped.")


if __name__ == "__main__":
    main()
