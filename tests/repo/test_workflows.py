from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load_workflow(name: str) -> dict[str, Any]:
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def _steps(workflow: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return workflow["jobs"][job]["steps"]


def _uses(steps: list[dict[str, Any]]) -> set[str]:
    return {step["uses"] for step in steps if "uses" in step}


def _run_commands(steps: list[dict[str, Any]]) -> str:
    return "\n".join(step["run"] for step in steps if "run" in step)


def _assert_no_continue_on_error(value: Any) -> None:
    if isinstance(value, dict):
        assert "continue-on-error" not in value
        for child in value.values():
            _assert_no_continue_on_error(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_continue_on_error(child)


def test_ci_has_exact_required_jobs_and_triggers() -> None:
    workflow = _load_workflow("ci.yml")

    assert workflow["name"] == "CI"
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert "pull_request" in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"]["NO_MKDOCS_2_WARNING"] == "true"
    assert set(workflow["jobs"]) == {"quality", "tests", "package", "docs"}
    assert all(job["runs-on"] == "ubuntu-latest" for job in workflow["jobs"].values())
    _assert_no_continue_on_error(workflow)


def test_ci_uses_current_actions_python_312_and_pip_cache() -> None:
    workflow = _load_workflow("ci.yml")

    for job_name in workflow["jobs"]:
        steps = _steps(workflow, job_name)
        assert "actions/checkout@v7" in _uses(steps)
        setup = next(step for step in steps if step.get("uses") == "actions/setup-python@v6")
        assert setup["with"] == {"python-version": "3.12", "cache": "pip"}
        assert "python -m pip install -e '.[dev,docs]'" in _run_commands(steps)


def test_ci_quality_and_test_commands_match_contributor_contract() -> None:
    workflow = _load_workflow("ci.yml")
    quality = _run_commands(_steps(workflow, "quality"))
    tests = _run_commands(_steps(workflow, "tests"))

    for command in (
        "python scripts/check_public_tree.py",
        "ruff format --check .",
        "ruff check .",
        "mypy src/online_gcs/config.py src/online_gcs/metrics.py src/online_gcs/cli.py",
    ):
        assert command in quality
    assert (
        "pytest tests/unit tests/repo --cov=online_gcs --cov-report=term-missing "
        "--cov-report=xml" in tests
    )
    assert 'pytest tests/integration tests/smoke -m "not slow and not visualization"' in tests


def test_ci_builds_verified_distributions_and_strict_docs() -> None:
    workflow = _load_workflow("ci.yml")
    package_steps = _steps(workflow, "package")
    docs = _run_commands(_steps(workflow, "docs"))

    package = _run_commands(package_steps)
    assert "python -m build" in package
    assert "python -m twine check dist/*" in package
    artifact = next(
        step for step in package_steps if step.get("uses") == "actions/upload-artifact@v7"
    )
    assert artifact["with"]["path"] == "dist/"
    assert "mkdocs build --strict" in docs


def test_ci_generates_coverage_xml_before_uploading_it() -> None:
    workflow = _load_workflow("ci.yml")
    steps = _steps(workflow, "tests")
    generation_index = next(
        index for index, step in enumerate(steps) if "--cov-report=xml" in step.get("run", "")
    )
    upload_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses") == "actions/upload-artifact@v7"
    )

    assert generation_index < upload_index
    assert steps[upload_index]["with"]["path"] == "coverage.xml"


def test_pages_deploys_successful_main_ci_with_minimum_permissions() -> None:
    workflow = _load_workflow("docs.yml")

    assert workflow["on"] == {
        "workflow_run": {"workflows": ["CI"], "types": ["completed"], "branches": ["main"]}
    }
    assert workflow["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    assert set(workflow["jobs"]) == {"deploy"}
    deploy = workflow["jobs"]["deploy"]
    deploy_gate = deploy["if"]
    for condition in (
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.head_repository.full_name == github.repository",
    ):
        assert condition in deploy_gate
    assert deploy["environment"]["name"] == "github-pages"

    steps = deploy["steps"]
    assert {
        "actions/checkout@v7",
        "actions/setup-python@v6",
        "actions/configure-pages@v6",
        "actions/upload-pages-artifact@v5",
        "actions/deploy-pages@v5",
    }.issubset(_uses(steps))
    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v7")
    assert checkout["with"]["ref"] == "${{ github.event.workflow_run.head_sha }}"
    setup = next(step for step in steps if step.get("uses") == "actions/setup-python@v6")
    assert setup["with"] == {"python-version": "3.12", "cache": "pip"}
    commands = _run_commands(steps)
    assert "python -m pip install -e '.[dev,docs]'" in commands
    assert "mkdocs build --strict" in commands
    upload = next(step for step in steps if step.get("uses") == "actions/upload-pages-artifact@v5")
    assert upload["with"]["path"] == "site/"
    _assert_no_continue_on_error(workflow)
