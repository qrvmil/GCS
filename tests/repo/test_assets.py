from pathlib import Path
import subprocess
import struct
import sys

import pytest

from scripts import capture_demo


SCENE_NAMES = ("single-shelf.png", "two-shelves.png", "three-shelves.png")


def test_scene_assets_are_valid_nontrivial_pngs() -> None:
    for name in SCENE_NAMES:
        path = Path("docs/assets/scenes") / name
        assert path.is_file()
        assert 100_000 < path.stat().st_size < 1_500_000

        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert data[12:16] == b"IHDR"
        width, height = struct.unpack(">II", data[16:24])
        assert width > 0
        assert height > 0


def test_cleanup_removes_only_numbered_capture_frames(tmp_path: Path) -> None:
    assert hasattr(capture_demo, "cleanup_numbered_frames")
    removable = ("frame-00.png", "frame-09.png", "frame-100.png")
    preserved = ("frame-final.png", "scene.png", "notes.txt")
    for name in (*removable, *preserved):
        (tmp_path / name).write_bytes(b"old")

    capture_demo.cleanup_numbered_frames(tmp_path)

    assert all(not (tmp_path / name).exists() for name in removable)
    assert all((tmp_path / name).is_file() for name in preserved)


def test_ffmpeg_command_limits_input_to_requested_frames() -> None:
    command = capture_demo.ffmpeg_command(Path("artifacts/demo"), frames=5)

    assert "-i artifacts/demo/frame-%02d.png" in command
    assert "-frames:v 5" in command
    assert command.endswith("docs/assets/online-expansion.gif")


@pytest.mark.parametrize(
    "url",
    (
        "http://localhost:7001",
        "http://127.0.0.1:7001",
        "http://[::1]:7001",
    ),
)
def test_capture_url_accepts_only_loopback_hosts(url: str) -> None:
    assert hasattr(capture_demo, "validate_capture_url")
    capture_demo.validate_capture_url(url)


@pytest.mark.parametrize(
    "url",
    (
        "https://example.com/meshcat",
        "file:///tmp/meshcat.html",
        "javascript:alert(1)",
    ),
)
def test_capture_url_rejects_remote_or_unsafe_urls(url: str) -> None:
    assert hasattr(capture_demo, "validate_capture_url")
    with pytest.raises(ValueError, match="loopback"):
        capture_demo.validate_capture_url(url)


def test_capture_demo_exposes_reproducible_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/capture_demo.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for flag in (
        "--url",
        "--output-dir",
        "--width",
        "--height",
        "--frames",
        "--browser-executable",
        "--camera-zoom-steps",
    ):
        assert flag in result.stdout
