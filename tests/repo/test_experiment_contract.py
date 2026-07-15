import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


EXPERIMENTS = sorted(Path("experiments").glob("*.py"))
LEGACY_IMPORT_ROOTS = {"helpers", "experiments.utils"}


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=str)
def test_experiments_do_not_import_legacy_modules(path: Path) -> None:
    imports = []
    for node in ast.walk(_module(path)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert not any(
        name == root or name.startswith(f"{root}.")
        for name in imports
        for root in LEGACY_IMPORT_ROOTS
    )


@pytest.mark.parametrize("path", EXPERIMENTS, ids=str)
def test_experiment_output_defaults_are_ignored(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    output_defaults = [
        keyword.value.value
        for node in ast.walk(_module(path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--output"
        for keyword in node.keywords
        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant)
    ]

    if path.stem == "__init__":
        assert output_defaults == []
    else:
        assert output_defaults
        assert all(str(default).startswith("artifacts/") for default in output_defaults), source


@pytest.mark.parametrize("path", EXPERIMENTS, ids=str)
def test_experiments_do_not_patch_import_paths(path: Path) -> None:
    source = path.read_text(encoding="utf-8")

    assert "sys.path" not in source
    assert "spec_from_file_location" not in source


def test_importing_experiments_does_not_run_them(tmp_path: Path) -> None:
    repository = Path.cwd()
    modules = [f"experiments.{path.stem}" for path in EXPERIMENTS if path.stem != "__init__"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository)
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    subprocess.run(
        [sys.executable, "-c", "; ".join(f"import {module}" for module in modules)],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    assert not (tmp_path / "artifacts").exists()
