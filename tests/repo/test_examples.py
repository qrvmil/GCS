import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    ["minimal_online.py", "visualize_single_shelf.py"],
)
def test_examples_use_only_the_public_package_api(name: str) -> None:
    path = Path("examples") / name
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = [node for node in module.body if isinstance(node, ast.ImportFrom)]

    assert [(node.module, [alias.name for alias in node.names]) for node in imports] == [
        ("online_gcs", ["OnlineGCS", "SceneType"])
    ]


def test_examples_readme_documents_both_commands() -> None:
    text = Path("examples/README.md").read_text(encoding="utf-8")

    assert "Python 3.12" in text
    assert "python examples/minimal_online.py" in text
    assert "python examples/visualize_single_shelf.py" in text
    assert "Ctrl+C" in text
