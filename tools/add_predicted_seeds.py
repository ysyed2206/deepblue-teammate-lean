"""Add analyzed-but-never-observed positions to the opening book.

Companion to build_opening_book.py, which only analyzes positions we have
actually been assigned. This handles the other half: positions we predict
are LIKELY to be assigned, based on the dominant family identified across
the 28 observed seeds (a Closed Sicilian skeleton -- 1.e4 c5 2.Nc3 Nc6 3.g3
g6 4.Bg2 Bg7 5.d3 and its natural branches -- appearing in 7+ of 28 seeds).
Each candidate line is a real, legal opening verified via python-chess
before being queued here; entries are marked "predicted": true so the
provenance is never confused with an actually-observed seed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastsearch57  # noqa: E402
from deepblue.fastcore import from_fen  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", type=Path, required=True,
                     help="JSON list of full FENs to analyze and add")
    ap.add_argument("--out", type=Path,
                     default=Path(r"C:\Users\uniqu\Downloads\deepblue-teammate-lean\deepblue\opening_book.json"))
    ap.add_argument("--soft-ms", type=float, default=90_000.0)
    ap.add_argument("--hard-ms", type=float, default=110_000.0)
    args = ap.parse_args()

    fens = json.loads(args.fens.read_text(encoding="utf-8"))
    book = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}

    fastsearch57.warm_up()

    added = 0
    for i, full_fen in enumerate(fens):
        key = " ".join(full_fen.split(" ")[:4])
        if key in book:
            print(f"[{i+1}/{len(fens)}] already in book, skipping")
            continue
        started = time.monotonic()
        engine = fastsearch57.FastEngine57()
        move, score, depth, nodes, elapsed_ms = engine.search(
            from_fen(full_fen), args.soft_ms, args.hard_ms)
        book[key] = {
            "rounds_seen": [],
            "predicted": True,
            "best_move": move,
            "score": score,
            "depth": depth,
            "nodes": nodes,
            "analysis_ms": elapsed_ms,
        }
        args.out.write_text(json.dumps(book, indent=2) + "\n", encoding="utf-8")
        added += 1
        print(f"[{i+1}/{len(fens)}] move={move} score={score} depth={depth} "
              f"nodes={nodes:,} ({time.monotonic()-started:.0f}s)", flush=True)

    print(f"\nadded {added} predicted positions; book now has {len(book)} total entries")


if __name__ == "__main__":
    main()
