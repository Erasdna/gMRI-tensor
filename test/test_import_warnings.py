import os
import subprocess
import sys
from pathlib import Path


def test_import_emits_no_tensorly_syntax_warnings(tmp_path: Path) -> None:
    # An empty bytecode cache forces recompilation, as in a read-only container.
    env = {**os.environ, "PYTHONPYCACHEPREFIX": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-W", "default", "-c", "import gMRItensor"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "SyntaxWarning" not in result.stderr, result.stderr
