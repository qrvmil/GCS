import re
from pathlib import Path


DOC_PAGES = (
    "index.md",
    "installation.md",
    "quickstart.md",
    "architecture.md",
    "api.md",
    "cli.md",
    "reproducibility.md",
    "troubleshooting.md",
)


def headings(path: str | Path) -> list[int]:
    text = Path(path).read_text(encoding="utf-8")
    return [len(match.group(1)) for match in re.finditer(r"^(#{1,6}) ", text, re.M)]


def fenced_code(path: str | Path) -> list[tuple[str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    return re.findall(r"```([^\n]*)\n(.*?)```", text, re.S)


def image_targets(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return re.findall(r"!\[[^]]*]\(([^)]+)\)", text)


def test_readmes_share_structure_commands_and_images() -> None:
    assert headings("README.md") == headings("README.ru.md")
    assert fenced_code("README.md") == fenced_code("README.ru.md")
    assert image_targets("README.md") == image_targets("README.ru.md")


def test_language_switches_are_reciprocal() -> None:
    assert "README.ru.md" in Path("README.md").read_text(encoding="utf-8")
    assert "README.md" in Path("README.ru.md").read_text(encoding="utf-8")


def test_documentation_pages_have_translated_twins() -> None:
    for page in DOC_PAGES:
        english = Path("docs/en") / page
        russian = Path("docs/ru") / page
        assert english.is_file(), english
        assert russian.is_file(), russian
        assert headings(english) == headings(russian), page
        assert fenced_code(english) == fenced_code(russian), page
        assert image_targets(english) == image_targets(russian), page


def test_standard_run_baselines_are_disclosed_in_both_languages() -> None:
    english_pages = (
        "README.md",
        "docs/en/index.md",
        "docs/en/quickstart.md",
        "docs/en/architecture.md",
    )
    russian_pages = (
        "README.ru.md",
        "docs/ru/index.md",
        "docs/ru/quickstart.md",
        "docs/ru/architecture.md",
    )
    for page in english_pages:
        text = Path(page).read_text(encoding="utf-8")
        assert "every query" in text, page
        assert "TrajOpt" in text, page
    for page in russian_pages:
        text = Path(page).read_text(encoding="utf-8")
        assert "каждого запроса" in text, page
        assert "TrajOpt" in text, page


def test_cli_output_example_uses_ignored_artifacts_directory() -> None:
    expected = (
        "bash",
        (
            "mkdir -p artifacts\n"
            "online-gcs --scene SINGLE_SHELF --iterations 1 "
            "--output artifacts/regions.yaml\n"
        ),
    )
    for page in ("docs/en/cli.md", "docs/ru/cli.md"):
        assert expected in fenced_code(page), page


def test_pipeline_shows_baselines_and_continuous_retry_connector() -> None:
    svg = Path("docs/assets/pipeline.svg").read_text(encoding="utf-8")
    assert "RRT* + TrajOpt" in svg
    assert 'd="M850 300 H620"' in svg
    assert 'd="M850 300 H790"' not in svg
