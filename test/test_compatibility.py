"""The package must run on Python 3.10, the oldest `requires-python`
allows, although development uses a newer one."""
import ast
from pathlib import Path

import pytest

SOURCES = sorted((Path(__file__).parents[1] / "src").rglob("*.py"))


def _starred_subscripts(tree: ast.AST) -> list[int]:
    """Lines with `x[*a]`, Python 3.11 syntax that `ast.parse`'s
    `feature_version` does not reject."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            index = node.slice
            parts = index.elts if isinstance(index, ast.Tuple) else [index]
            if any(isinstance(part, ast.Starred) for part in parts):
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_sources_use_python_310_syntax(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path), feature_version=(3, 10))

    assert _starred_subscripts(tree) == [], f"{path}: x[*a] needs Python 3.11"


def test_the_check_finds_a_starred_subscript() -> None:
    assert _starred_subscripts(ast.parse("x[*idx.T] = 1")) == [1]
