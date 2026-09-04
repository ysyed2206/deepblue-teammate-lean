"""Differential test for the qsearch tactical-only generator.

Compares ``generate_pseudo_tactical`` against filtering the already-verified
full pseudo-legal generator to captures or promotions.  Random walks use the
engine's exact legal-move filter, so EP, promotions and unusual sparse positions
naturally enter the sample.  This test has no python-chess dependency and can
run inside the submission environment or a stripped development container.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepblue import fastcore as F  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def is_tactical(move: int) -> bool:
    captured = (move >> F.CAPTURE_SHIFT) & 15
    promotion = (move >> F.PROMOTION_SHIFT) & 15
    return captured != F.NO_PIECE or promotion != F.NO_PIECE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-walk", type=int, default=160)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    full = np.zeros(F.MAX_MOVES, dtype=np.uint32)
    tactical = np.zeros(F.MAX_MOVES, dtype=np.uint32)
    legal = np.zeros((F.MAX_PLY, F.MAX_MOVES), dtype=np.uint32)
    pseudo = np.zeros((F.MAX_PLY, F.MAX_MOVES), dtype=np.uint32)
    undo = np.zeros(F.MAX_PLY * F.UNDO_STRIDE, dtype=np.int64)

    bb, occ, mail, st = F.from_fen(START)
    checked = 0
    max_tactical = 0
    for position_index in range(args.positions):
        n_full = F.generate_pseudo_legal(bb, occ, mail, st, full)
        expected = {int(full[i]) for i in range(n_full) if is_tactical(int(full[i]))}
        n_tac = F.generate_pseudo_tactical(bb, occ, mail, st, tactical)
        actual = {int(tactical[i]) for i in range(n_tac)}
        if expected != actual:
            print(f"FAIL at sample {position_index}")
            print("missing:", sorted(expected - actual)[:20])
            print("extra:  ", sorted(actual - expected)[:20])
            raise SystemExit(1)
        if n_tac != len(actual):
            print(f"FAIL duplicate tactical moves at sample {position_index}")
            raise SystemExit(1)
        max_tactical = max(max_tactical, n_tac)
        checked += 1

        # Advance by a random legal move. Restart periodically or at game end.
        legal_count = F.generate_legal(bb, occ, mail, st, legal, pseudo, undo, 0)
        if legal_count == 0 or position_index % args.max_walk == args.max_walk - 1:
            bb, occ, mail, st = F.from_fen(START)
            continue
        move = legal[0, rng.randrange(legal_count)]
        F.make_move(bb, occ, mail, st, move, undo, 0)

    print(f"tactical differential: 0 failures / {checked:,} positions")
    print(f"maximum tactical pseudo-moves observed: {max_tactical} / {F.MAX_MOVES}")


if __name__ == "__main__":
    main()
