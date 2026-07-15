import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.check_public_tree import find_forbidden_tracked


class PublicTreeTest(unittest.TestCase):
    def test_current_git_tree_contains_no_research_artifacts(self) -> None:
        tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        self.assertEqual(find_forbidden_tracked(tracked), [])

    def test_policy_accepts_documentation_assets(self) -> None:
        self.assertEqual(find_forbidden_tracked(["docs/assets/scenes/single-shelf.png"]), [])

    def test_policy_rejects_generated_data(self) -> None:
        paths = ["experiments/results/run.csv", "docs/report.pdf", "run.log"]
        self.assertEqual(find_forbidden_tracked(paths), paths)

    def test_generated_coverage_report_is_ignored(self) -> None:
        result = subprocess.run(["git", "check-ignore", "--quiet", "coverage.xml"], check=False)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
