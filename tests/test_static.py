"""Static analysis and bytecode compilation test.

Ensures that all source files and scripts compile cleanly without any syntax errors,
invalid future imports, or compilation issues.
"""

from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def test_compileall_src_and_scripts():
    """Verify compileall succeeds on src and scripts directories."""
    src_dir = str(BASE_DIR / "src")
    scripts_dir = str(BASE_DIR / "scripts")

    # 1. Standard library compileall module verification
    ok_src = compileall.compile_dir(src_dir, force=True, quiet=1)
    assert ok_src, "compileall failed on src directory"

    ok_scripts = compileall.compile_dir(scripts_dir, force=True, quiet=1)
    assert ok_scripts, "compileall failed on scripts directory"

    # 2. Subprocess command line verification
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", src_dir, scripts_dir],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"compileall command failed:\n{result.stderr}\n{result.stdout}"


def test_pyflakes_clean():
    """Verify pyflakes reports zero syntax errors, undefined names, or unused imports."""
    src_dir = str(BASE_DIR / "src")
    scripts_dir = str(BASE_DIR / "scripts")
    targets = [src_dir, scripts_dir]
    app_file = BASE_DIR / "app.py"
    if app_file.exists():
        targets.append(str(app_file))

    result = subprocess.run(
        [sys.executable, "-m", "pyflakes", *targets],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"pyflakes found issues:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip() == "", f"pyflakes output should be empty, got:\n{result.stdout}"
