import subprocess
import sys

from online_gcs.cli import main


def test_zero_iterations_returns_usage_error(capsys) -> None:
    assert main(["--iterations", "0"]) == 2
    assert "iterations must be at least 1" in capsys.readouterr().err


def test_help_does_not_construct_drake_scene() -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_importing_cli_does_not_import_drake_or_online_planner() -> None:
    script = """
import sys
from online_gcs.cli import build_parser, main

assert "pydrake" not in sys.modules
assert "online_gcs.planners.online" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
