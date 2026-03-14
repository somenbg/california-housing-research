"""Git operations for experiment branch management."""

import subprocess
from typing import Optional


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_current_branch() -> str:
    r = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return r.stdout.strip()


def get_short_hash() -> str:
    r = _run(["git", "rev-parse", "--short=7", "HEAD"])
    return r.stdout.strip()


def branch_exists(name: str) -> bool:
    r = _run(["git", "rev-parse", "--verify", name], check=False)
    return r.returncode == 0


def create_branch(name: str):
    _run(["git", "checkout", "-b", name])


def checkout(name: str):
    _run(["git", "checkout", name])


def commit(message: str, files: Optional[list[str]] = None):
    """Stage files and commit. If files is None, stage all changes."""
    if files:
        _run(["git", "add"] + files)
    else:
        _run(["git", "add", "-A"])
    _run(["git", "commit", "-m", message])


def reset_hard(ref: str = "HEAD~1"):
    _run(["git", "reset", "--hard", ref])


def get_diff(ref: str = "HEAD~1") -> str:
    r = _run(["git", "diff", ref, "HEAD"], check=False)
    return r.stdout


def is_clean() -> bool:
    r = _run(["git", "status", "--porcelain"])
    return len(r.stdout.strip()) == 0
