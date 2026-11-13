"""Every `python -m src.…` command documented in README.md exists and parses its arguments."""

import re
import subprocess
import sys

import pytest

from src.core.paths import REPO_ROOT

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
COMMANDS = sorted(set(re.findall(r"^python -m (src[\w.]+)(.*)$", README, flags=re.M)))


def test_readme_documents_commands():
    assert len(COMMANDS) >= 8


@pytest.mark.parametrize("module, args", COMMANDS, ids=[m for m, _ in COMMANDS])
def test_readme_command_parses(module, args):
    r = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    for flag in re.findall(r"(--[\w-]+)", args):
        assert flag in r.stdout, f"{module} --help does not mention {flag} used in README"
