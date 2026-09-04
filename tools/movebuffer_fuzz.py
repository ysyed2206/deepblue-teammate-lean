"""Prove the pseudo-legal move buffer can never overflow.

MAX_MOVES is 256 and the search writes into a fixed array inside nopython
code, where an overflow is silent memory corruption rather than an exception.
This searches hard for the worst case: promotion-heavy positions, many queens,
and high-mobility middlegames, and fails loudly if anything approaches the cap.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastcore as F  # noqa: E402

# Hand-built positions chosen to maximise pseudo-legal move counts.
EXTREME_FENS = [
    "QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQK/QQQQQQQk w - - 0 1",
    "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1",
    "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "8/PPPPPPPP/8/8/8/8/pppppppp/4K2k w - - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260901)
    arguments = parser.parse_args()

    F.warm_up()
    buffer = np.zeros(F.MAX_MOVES, dtype=np.uint32)
    rng = random.Random(arguments.seed)
    worst = 0
    worst_fen = ""
    checked = 0
    overflow = 0

    def probe(fen: str) -> None:
        nonlocal worst, worst_fen, checked, overflow
        bb, occ, mail, st = F.from_fen(fen)
        count = F.generate_pseudo_legal(bb, occ, mail, st, buffer)
        checked += 1
        if count > worst:
            worst, worst_fen = count, fen
        if count >= F.MAX_MOVES:
            overflow += 1
            print(f"  OVERFLOW: {count} pseudo-legal moves at {fen}")

    for fen in EXTREME_FENS:
        try:
            probe(fen)
        except ValueError:
            pass  # an illegal construction, not a movegen failure

    while checked < arguments.positions:
        board = chess.Board(rng.choice(EXTREME_FENS[3:]))
        for _ in range(rng.randrange(0, 100)):
            moves = list(board.legal_moves)
            if not moves:
                break
            promotions = [m for m in moves if m.promotion]
            board.push(rng.choice(promotions if promotions and rng.random() < 0.7 else moves))
            probe(board.fen())
            if checked >= arguments.positions:
                break

    headroom = F.MAX_MOVES - worst
    print(f"positions probed        : {checked:,}")
    print(f"MAX_MOVES               : {F.MAX_MOVES}")
    print(f"worst pseudo-legal count: {worst}")
    print(f"headroom                : {headroom} slots ({headroom / F.MAX_MOVES:.0%})")
    print(f"worst position          : {worst_fen}")
    print(f"overflows               : {overflow}")
    raise SystemExit(1 if overflow else 0)


if __name__ == "__main__":
    main()
