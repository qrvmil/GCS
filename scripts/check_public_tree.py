from collections.abc import Iterable
from pathlib import PurePosixPath
import subprocess
import sys

FORBIDDEN_SUFFIXES = {
    ".aux", ".bbl", ".bcf", ".blg", ".csv", ".doc", ".docm", ".docx",
    ".fdb_latexmk", ".fls", ".ipynb", ".log", ".pdf", ".run.xml", ".synctex.gz", ".tex", ".toc",
}
FORBIDDEN_PARTS = {"__pycache__", ".cache", ".matplotlib-cache", "results", "logs"}


def find_forbidden_tracked(paths: Iterable[str]) -> list[str]:
    forbidden: list[str] = []
    for value in paths:
        path = PurePosixPath(value)
        joined_suffix = "".join(path.suffixes[-2:])
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or joined_suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(value)
        elif any(part in FORBIDDEN_PARTS for part in path.parts):
            forbidden.append(value)
    return forbidden


def main() -> int:
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    forbidden = find_forbidden_tracked(tracked)
    if forbidden:
        for path in forbidden:
            print(path)
        return 1
    print("Public tree policy: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
