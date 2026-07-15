import ast
from pathlib import Path

import pytest


EXPERIMENTS = sorted(Path("experiments").glob("*.py"))
LEGACY_IMPORT_ROOTS = {"helpers", "experiments.utils"}


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_main_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _is_main_call(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "main"
        and not node.value.args
        and not node.value.keywords
    )


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


@pytest.mark.parametrize(
    "path",
    [path for path in EXPERIMENTS if path.stem != "__init__"],
    ids=str,
)
def test_experiments_have_guarded_main_entrypoints(path: Path) -> None:
    module = _module(path)
    main_functions = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    main_guards = [node for node in module.body if _is_main_guard(node)]

    assert len(main_functions) == 1
    assert len(main_guards) == 1
    assert len(main_guards[0].body) == 1
    assert _is_main_call(main_guards[0].body[0])
    assert main_guards[0].orelse == []


@pytest.mark.parametrize("path", EXPERIMENTS, ids=str)
def test_experiments_have_no_unguarded_top_level_execution(path: Path) -> None:
    safe_statements = (
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.FunctionDef,
        ast.Import,
        ast.ImportFrom,
    )

    for node in _module(path).body:
        if isinstance(node, safe_statements) or _is_main_guard(node):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        pytest.fail(f"unguarded top-level {type(node).__name__} in {path}:{node.lineno}")
