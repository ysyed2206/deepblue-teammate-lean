"""Run the permanent regression corpus against any fastsearchN variant."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastsearch as fs
from deepblue.fastcore import from_fen
from tools.regression import CORPUS, check


class VariantEngine:
    def __init__(self, name: str) -> None:
        mod = importlib.import_module(f"deepblue.{name}")
        suffix = name.removeprefix("fastsearch") or ""
        cls = getattr(mod, "FastEngine" + suffix)
        mod.warm_up()
        self.name = name
        self._engine = cls()

    def search(self, fen: str, movetime_ms: int):
        move, score, depth, _, _ = self._engine.search(
            from_fen(fen), movetime_ms, movetime_ms * 1.5
        )
        return move, score, depth

    def insufficient(self, fen: str) -> bool:
        bb, occ, _, _ = from_fen(fen)
        return bool(fs.insufficient_material(bb, occ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", help="e.g. fastsearch12")
    ap.add_argument("--movetime-ms", type=int, default=400)
    args = ap.parse_args()

    engine = VariantEngine(args.variant)
    print(f"engine: {engine.name}\n")
    failures = 0
    total = 0
    for line in CORPUS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tag, fen, note = (part.strip() for part in line.split("|", 2))
        total += 1
        passed, detail = check(engine, tag, fen, note, args.movetime_ms)
        failures += 0 if passed else 1
        print(f"  {'pass' if passed else 'FAIL'}  {tag:9s} {detail}")
        if not passed:
            print(f"        fen  {fen}")
            print(f"        note {note}")
    print(f"\n{total - failures}/{total} regression positions passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
