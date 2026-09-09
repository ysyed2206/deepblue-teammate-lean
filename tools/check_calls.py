"""Verify every recursive negamax/search_root call matches its definition.

WHY. Three separate builds this session shipped a call site with the wrong
argument count, because the build scripts patch on exact text and this codebase
formats the same argument list several different ways depending on indentation.
An `assert count == N` proves the token I CHOSE appears N times; it cannot tell
me about sites that token never reaches. Numba catches the mismatch, but only
after a full compile -- typically a minute or two, and in one case only after a
match had already been launched and wasted a run.

This parses the module and compares each call's argument count against the
definition, in about a second.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def check(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs = {n.name: len(n.args.args) for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)}
    problems = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in defs or node.keywords:
            continue
        want, got = defs[name], len(node.args)
        if want != got:
            print(f"  line {node.lineno:5d}  {name}() takes {want} args, called with {got}")
            problems += 1
    return problems


if __name__ == "__main__":
    total = 0
    for arg in sys.argv[1:]:
        path = Path(arg)
        bad = check(path)
        print(f"{path.name}: {'OK' if not bad else str(bad) + ' MISMATCH(ES)'}")
        total += bad
    sys.exit(1 if total else 0)
