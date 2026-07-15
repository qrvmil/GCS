from pathlib import Path

import yaml


def test_license_is_mit_and_names_copyright_holder() -> None:
    text = Path("LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Copyright (c) 2026 Milana Krivova" in text


def test_citation_matches_release() -> None:
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))
    assert citation["version"] == "0.1.0"
    assert citation["repository-code"] == "https://github.com/qrvmil/GCS"
    assert citation["license"] == "MIT"


def test_typed_package_marker_is_present() -> None:
    assert Path("src/online_gcs/py.typed").is_file()
