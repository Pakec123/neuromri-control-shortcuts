"""Regression: `python -m neuromri.<p2 module>` must not run fit functions from __main__ (joblib/loky pickling)."""
import ast
from pathlib import Path
import pytest

@pytest.mark.parametrize("mod", ["p2_threshold_transport", "p2_external"])
def test_main_calls_package_run(mod):
    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath("neuromri", f"{mod}.py").read_text(encoding="utf-8"))
    main = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
    assert main, mod
    src = ast.unparse(main[0])
    assert f"from neuromri.{mod} import run" in src and "_run(" in src


def test_provenance_git_fallback_matches_git():
    import shutil, subprocess
    from neuromri.common import git_commit_from_files, source_fingerprint
    if shutil.which("git"):
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        assert git_commit_from_files() == head
    fp = source_fingerprint()
    assert len(fp["sha256"]) == 64 and fp["files"] > 10
