#!/usr/bin/env python3
"""Capture reproducible screenshots from a running Meshcat demo."""

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import re
import shlex
import sys
import time
from urllib.parse import urlparse


DEFAULT_CHROME_MACOS = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
CHROME_CANDIDATES = (
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/google-chrome-stable"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
)
NUMBERED_FRAME = re.compile(r"frame-\d+\.png")
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def build_parser() -> ArgumentParser:
    """Build the command-line parser without importing the optional dependency."""
    parser = ArgumentParser(
        description="Capture fixed-size frames from an already-running Meshcat URL."
    )
    parser.add_argument("--url", required=True, help="Meshcat URL printed by online-gcs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/demo"),
        help="ignored directory for frame-00.png, frame-01.png, ...",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument(
        "--browser-executable",
        type=Path,
        default=None,
        help="system Chromium/Chrome executable; auto-detected when omitted",
    )
    parser.add_argument(
        "--camera-zoom-steps",
        type=int,
        default=3,
        help="deterministic inward mouse-wheel steps applied after Meshcat loads",
    )
    parser.add_argument("--initial-wait-seconds", type=float, default=2.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    return parser


def resolve_browser_executable(explicit: Path | None) -> Path | None:
    """Return a usable system browser, or None for Playwright's bundled browser."""
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f"browser executable does not exist: {explicit}")
        return explicit

    environment_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    candidates = (
        *((Path(environment_path),) if environment_path else ()),
        *((DEFAULT_CHROME_MACOS,) if sys.platform == "darwin" else ()),
        *CHROME_CANDIDATES,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def validate_args(args: Namespace) -> None:
    """Reject parameters that cannot produce a meaningful capture."""
    for name in ("width", "height", "frames"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")
    if args.camera_zoom_steps < 0:
        raise ValueError("--camera-zoom-steps cannot be negative")
    for name in ("initial_wait_seconds", "interval_seconds"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")


def validate_capture_url(url: str) -> None:
    """Allow capture only from an HTTP(S) Meshcat server on the local machine."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError("--url must use HTTP(S) and a loopback host")


def cleanup_numbered_frames(output_dir: Path) -> int:
    """Remove prior numbered capture frames without touching unrelated files."""
    removed = 0
    for path in output_dir.iterdir():
        if path.is_file() and NUMBERED_FRAME.fullmatch(path.name):
            path.unlink()
            removed += 1
    return removed


def ffmpeg_command(output_dir: Path, frames: int) -> str:
    """Return the exact command used to assemble the documented GIF."""
    input_pattern = shlex.quote(str(output_dir / "frame-%02d.png"))
    output = shlex.quote("docs/assets/online-expansion.gif")
    return (
        f"ffmpeg -y -framerate 2 -i {input_pattern} "
        f'-vf "scale=960:-2:flags=lanczos" -frames:v {frames} -loop 0 {output}'
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        validate_capture_url(args.url)
        executable = resolve_browser_executable(args.browser_executable)
    except ValueError as exc:
        print(f"capture-demo: error: {exc}", file=sys.stderr)
        return 2

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "capture-demo: install the demo extra with `python -m pip install -e '.[demo]'`",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser_args = []
        if sys.platform == "darwin":
            browser_args.append("--use-angle=metal")
        launch_options = {"headless": True, "args": browser_args}
        if executable is not None:
            launch_options["executable_path"] = str(executable)
        browser = playwright.chromium.launch(**launch_options)
        try:
            removed = cleanup_numbered_frames(args.output_dir)
            if removed:
                print(f"Removed {removed} stale numbered frame(s)")
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_selector("canvas", state="visible", timeout=30_000)
            time.sleep(args.initial_wait_seconds)
            page.mouse.move(args.width / 2, args.height / 2)
            for _ in range(args.camera_zoom_steps):
                page.mouse.wheel(0, -200)
                page.wait_for_timeout(100)
            for index in range(args.frames):
                page.evaluate("window.dispatchEvent(new Event('resize'))")
                page.mouse.move(args.width / 2 + index % 2, args.height / 2)
                page.wait_for_timeout(250)
                path = args.output_dir / f"frame-{index:02d}.png"
                page.screenshot(path=path)
                print(f"Captured {path}")
                if index + 1 < args.frames:
                    time.sleep(args.interval_seconds)
        finally:
            browser.close()

    print(ffmpeg_command(args.output_dir, args.frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
