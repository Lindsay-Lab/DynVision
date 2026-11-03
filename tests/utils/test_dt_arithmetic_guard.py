"""Guard: no unit conversion with ``dt`` outside ``dynvision.utils.timing``.

Every millisecond <-> step conversion goes through ``utils.timing``, which is
exact or raises. A bare ``value / dt`` or ``index * dt`` elsewhere is how the
codebase drifted into three rounding modes and one hard-coded plotting ``dt``,
so this test fails on any new one.

The scan is syntactic: it flags ``*``, ``/``, ``//`` and ``%`` with ``dt`` (a
name, an attribute ``.dt``, or either wrapped in ``float()``/``int()``) as an
operand, and reports the enclosing function.

Two kinds of site are handled separately:

- ``EXEMPT``: uses of ``dt`` that are not unit conversions and stay by design.
- ``ALLOWLIST``: conversion sites that predate the rule. Each implementation
  item of the temporal-units plan removes the entries it fixes, and the test
  fails on a stale entry, so the list can only shrink. The plan is complete
  when it is empty.
"""

import ast
import subprocess
from pathlib import Path
from typing import Iterator, List, Set, Tuple

PACKAGE = Path(__file__).resolve().parents[2] / "dynvision"

#: Files where dt is used for its own sake, not to convert units.
EXEMPT_FILES: frozenset[str] = frozenset(
    {
        # The single conversion authority.
        "dynvision/utils/timing.py",
        # The integration step itself: x += dt / tau * (...).
        "dynvision/model_components/dynamics_solver.py",
    }
)

#: Functions where dt is used for its own sake, not to convert units.
EXEMPT_SITES: frozenset[str] = frozenset(
    {
        # dt / tau diagnostics: tau is a continuous time constant, not a window.
        "dynvision/params/model_params.py::ModelParams.stability_ratio",
        "dynvision/params/model_params.py::ModelParams.effective_tau_steps",
    }
)

#: Conversion sites that predate the rule. Remove entries as they are fixed.
ALLOWLIST: frozenset[str] = frozenset()

_ARITHMETIC = (ast.Div, ast.Mult, ast.FloorDiv, ast.Mod)


def _is_dt(node: ast.AST) -> bool:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("float", "int")
        and node.args
    ):
        return _is_dt(node.args[0])
    if isinstance(node, ast.Name):
        return node.id == "dt"
    if isinstance(node, ast.Attribute):
        return node.attr == "dt"
    return False


class _DtArithmetic(ast.NodeVisitor):
    def __init__(self) -> None:
        self._scope: list = []
        self.sites: Set[Tuple[str, int]] = set()

    def _visit_scope(self, node) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, _ARITHMETIC) and (_is_dt(node.left) or _is_dt(node.right)):
            self.sites.add((".".join(self._scope) or "<module>", node.lineno))
        self.generic_visit(node)


def _package_files() -> List[Path]:
    """The package's Python files as git tracks them.

    Only shipped code is checked: an ignored local scratch file (``temp_*``) is
    neither run nor reviewed. Without git, every file is scanned.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--", "*.py"],
            cwd=PACKAGE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return sorted(PACKAGE.rglob("*.py"))
    return sorted(PACKAGE / name for name in listed)


def _scan() -> Iterator[Tuple[str, int]]:
    """Yield ('path::qualname', line) for every dt arithmetic site."""
    for path in _package_files():
        relative = path.relative_to(PACKAGE.parent).as_posix()
        if relative in EXEMPT_FILES:
            continue
        visitor = _DtArithmetic()
        visitor.visit(ast.parse(path.read_text(), filename=str(path)))
        for qualname, line in visitor.sites:
            site = f"{relative}::{qualname}"
            if site not in EXEMPT_SITES:
                yield site, line


def test_no_new_dt_arithmetic():
    """A conversion outside utils.timing must go through it instead."""
    offenders = sorted(
        f"{site} (line {line})" for site, line in _scan() if site not in ALLOWLIST
    )
    assert not offenders, (
        "Unit arithmetic with dt outside dynvision/utils/timing.py; use "
        "to_timesteps()/to_ms() instead:\n  " + "\n  ".join(offenders)
    )


def test_allowlist_has_no_stale_entries():
    """A fixed site must leave the allowlist, so the list only shrinks."""
    found = {site for site, _ in _scan()}
    stale = sorted(ALLOWLIST - found)
    assert not stale, (
        "Allowlisted sites no longer convert with dt; remove them from "
        "ALLOWLIST:\n  " + "\n  ".join(stale)
    )


def test_the_scan_detects_conversions():
    """Self-check: the visitor flags the patterns the guard is for."""
    source = (
        "def f(t, dt, self):\n"
        "    a = int(t / dt)\n"
        "    b = index * self.dt\n"
        "    c = t // float(self.dt)\n"
        "    d = dt + 1\n"
    )
    visitor = _DtArithmetic()
    visitor.visit(ast.parse(source))
    assert sorted(line for _, line in visitor.sites) == [2, 3, 4]
