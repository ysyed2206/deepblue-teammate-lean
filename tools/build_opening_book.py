"""Deep-analyze every distinct seed position seen in real qualification games
and write the results to a small FEN-keyed lookup book.

Rationale (2026-09-05): every competition PGN reviewed carries a [SetUp]/
[FEN] header 6-8 moves deep, not move 1 -- and across 30 real rated games,
two pairs were BYTE-IDENTICAL seed positions from different rounds against
different opponents (round 1 == round 14, round 10 == round 23), plus two
more pairs one legal move apart (round 25 -> 26, round 9 -> 28). That is
confirmation, not speculation, that the tournament draws from a small reused
seed pool. Since the live per-move clock caps how deep the engine can look
in any one game, but nothing caps how long we can think about a position
OFFLINE before the qualification window closes, this pre-computes a much
deeper answer than live play could ever afford for positions we have
already been assigned at least once -- and, if the pool keeps recurring,
for positions we may be assigned again.

Safety: this is a pure lookup consulted BEFORE search, never a replacement
for it. Any position not in the book falls through to the engine exactly as
today. Wrong/stale book entries can only ever cost what a normal search
move would have cost anyway, since agent.py's own legal-move fallback and
repetition bookkeeping are unchanged and still apply on top of this.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastsearch57  # noqa: E402
from deepblue.fastcore import from_fen  # noqa: E402

HEADER_RE = re.compile(r'\[(Round|FEN)\s+"([^"]*)"\]')


def extract_unique_seeds(pgn_dir: Path) -> dict[str, list[str]]:
    """Return {normalized_fen_prefix: [round numbers]} for every distinct seed."""
    by_fen: dict[str, list[str]] = {}
    for path in sorted(pgn_dir.glob("*.pgn")):
        text = path.read_text(encoding="utf-8", errors="replace")
        round_no = fen = None
        for match in HEADER_RE.finditer(text):
            key, value = match.groups()
            if key == "Round":
                round_no = value
            elif key == "FEN":
                fen = value
        if fen is None:
            continue
        # Key on board+turn+castling+ep only -- halfmove/fullmove counters
        # don't change what the position IS, and two seeds that agree on
        # everything else but differ only there are the same tabiya.
        key = " ".join(fen.split(" ")[:4])
        by_fen.setdefault(key, []).append(round_no or path.stem)
    return by_fen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn-dir", type=Path, default=Path(r"C:\Users\uniqu\Downloads"))
    ap.add_argument("--out", type=Path,
                     default=Path(r"C:\Users\uniqu\Downloads\deepblue-teammate-lean\deepblue\opening_book.json"))
    ap.add_argument("--soft-ms", type=float, default=90_000.0)
    ap.add_argument("--hard-ms", type=float, default=110_000.0)
    args = ap.parse_args()

    seeds = extract_unique_seeds(args.pgn_dir)
    print(f"found {len(seeds)} distinct seed positions across the PGN pool", flush=True)

    fastsearch57.warm_up()

    book: dict[str, dict] = {}
    if args.out.exists():
        book = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"resuming: {len(book)} positions already analyzed", flush=True)

    for i, (key, rounds) in enumerate(sorted(seeds.items())):
        if key in book:
            continue
        # Reconstruct a full FEN (halfmove/fullmove counters don't affect
        # legality or the engine's move choice for a fresh search).
        full_fen = f"{key} 0 1"
        started = time.monotonic()
        engine = fastsearch57.FastEngine57()  # fresh TT/history per seed, no cross-contamination
        move, score, depth, nodes, elapsed_ms = engine.search(
            from_fen(full_fen), args.soft_ms, args.hard_ms)
        book[key] = {
            "rounds_seen": rounds,
            "best_move": move,
            "score": score,
            "depth": depth,
            "nodes": nodes,
            "analysis_ms": elapsed_ms,
        }
        args.out.write_text(json.dumps(book, indent=2) + "\n", encoding="utf-8")
        print(f"[{i+1}/{len(seeds)}] rounds {rounds}: move={move} score={score} "
              f"depth={depth} nodes={nodes:,} ({time.monotonic()-started:.0f}s)", flush=True)

    print(f"\nwrote {len(book)} positions to {args.out}")


if __name__ == "__main__":
    main()
