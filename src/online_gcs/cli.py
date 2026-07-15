"""Command-line interface for the online GCS planner."""

import argparse
import logging
import sys
from collections.abc import Sequence

from online_gcs.config import OnlineGCSConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without constructing a Drake scene."""
    parser = argparse.ArgumentParser(
        prog="online-gcs",
        description="Online GCS Region Building",
    )
    parser.add_argument(
        "--scene",
        default="SINGLE_SHELF",
        choices=["SINGLE_SHELF", "TWO_SHELVES", "TABLE_THREE_SHELVES"],
        help="Scene type (default: SINGLE_SHELF)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Max iterations (default: 50)",
    )
    parser.add_argument(
        "--keypoints",
        type=int,
        default=10,
        help="Number of keypoints per RRT path (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--prune-interval",
        type=int,
        default=20,
        help="Prune redundant regions every N iterations (default: 20, 0=never)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path for saving regions (optional)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable Meshcat visualization",
    )
    parser.add_argument(
        "--animation-speed",
        type=float,
        default=1.0,
        help="Animation speed multiplier (default: 1.0, higher=faster)",
    )
    parser.add_argument(
        "--smart-keypoints",
        action="store_true",
        help="Use smart keypoints",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel IRIS region exploration (background workers)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel exploration workers when --parallel (default: 2)",
    )
    parser.add_argument(
        "--k-shortest-paths",
        type=int,
        default=3,
        help="Number of shortest paths for GCS subgraph selection (default: 3)",
    )
    parser.add_argument(
        "--warmstart",
        action="store_true",
        help="Build initial IRIS regions before main loop (farthest-point sampling)",
    )
    parser.add_argument(
        "--warmstart-seeds",
        type=int,
        default=15,
        help="Number of seed points for warmstart IRIS regions (default: 15)",
    )
    parser.add_argument("--rrt-only", action="store_true", help="Run only RRT* path planning")
    parser.add_argument(
        "--opt-only",
        action="store_true",
        help="Run only optimization of RRT* path",
    )
    parser.add_argument(
        "--paper-figure",
        action="store_true",
        help="Build a single paper-friendly online expansion scene",
    )
    parser.add_argument(
        "--figure-target-idx",
        type=int,
        default=None,
        help="Optional target index to use for --paper-figure",
    )
    parser.add_argument(
        "--figure-hold-seconds",
        type=float,
        default=0.0,
        help="How long to keep the Meshcat figure scene open; 0 waits until Ctrl+C",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> OnlineGCSConfig:
    return OnlineGCSConfig(
        scene=args.scene,
        iterations=args.iterations,
        keypoints=args.keypoints,
        seed=args.seed,
        prune_interval=args.prune_interval,
        verbose=args.verbose,
        visualize=args.visualize,
        animation_speed=args.animation_speed,
        smart_keypoints=args.smart_keypoints,
        parallel=args.parallel,
        num_workers=args.num_workers,
        k_shortest_paths=args.k_shortest_paths,
        warmstart=args.warmstart,
        warmstart_seeds=args.warmstart_seeds,
    )


def _print_run_configuration(config: OnlineGCSConfig, args: argparse.Namespace) -> None:
    print("=" * 60)
    print("Online GCS Region Building")
    print("=" * 60)
    print(f"Scene:          {config.scene}")
    print(f"Iterations:     {config.iterations}")
    print(f"Keypoints:      {config.keypoints}")
    print(f"Random seed:    {config.seed}")
    print(f"Prune interval: {config.prune_interval}")
    print(f"Verbose:        {config.verbose}")
    print(f"Visualize:      {config.visualize}")
    parallel_suffix = f" ({config.num_workers} workers)" if config.parallel else ""
    print(f"Parallel:       {config.parallel}{parallel_suffix}")
    print(f"K shortest:     {config.k_shortest_paths}")
    warmstart_suffix = f" ({config.warmstart_seeds} seeds)" if config.warmstart else ""
    print(f"Warmstart:      {config.warmstart}{warmstart_suffix}")
    print(f"RRT only:       {args.rrt_only}")
    print(f"Opt only:       {args.opt_only}")
    print(f"Paper figure:   {args.paper_figure}")
    print("=" * 60)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate CLI arguments, run the selected planner mode, and return an exit code."""
    args = build_parser().parse_args(argv)

    try:
        config = _config_from_args(args)
        config.validate()

        from online_gcs.planners.online import OnlineGCS
        from online_gcs.scenes import SceneType

        logging.getLogger("drake").setLevel(logging.WARNING)
        scene_type = SceneType[config.scene]
        _print_run_configuration(config, args)

        online_gcs = OnlineGCS(
            scene_type=scene_type,
            random_seed=config.seed,
            logging=config.verbose,
            visualize=config.visualize,
            smart_keypoints=config.smart_keypoints,
            max_iterations=config.iterations,
            parallel_exploration=config.parallel,
            num_workers=config.num_workers,
            k_shortest_paths=config.k_shortest_paths,
            warmstart=config.warmstart,
            warmstart_seeds=config.warmstart_seeds,
        )

        if args.paper_figure:
            online_gcs.run_paper_figure_demo(
                num_keypoints=config.keypoints,
                target_idx=args.figure_target_idx,
                hold_seconds=args.figure_hold_seconds,
            )
        elif args.rrt_only:
            online_gcs.run_rrt_only()
        elif args.opt_only:
            online_gcs.run_opt_only()
        else:
            online_gcs.run(
                num_keypoints=config.keypoints,
                prune_interval=config.prune_interval,
                animation_speed=config.animation_speed,
            )

        if args.output:
            online_gcs.save_regions(args.output)
    except (ValueError, RuntimeError) as exc:
        print(f"online-gcs: error: {exc}", file=sys.stderr)
        return 2

    return 0
