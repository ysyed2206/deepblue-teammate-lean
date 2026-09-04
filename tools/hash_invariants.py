"""Permanent invariants for Deep Blue's canonical Zobrist hash.

The key intentionally follows python-chess/FIDE repetition identity: a nominal
en-passant square contributes only when at least one *legal* en-passant capture
exists.  This tool has no python-chess dependency; the three targeted EP cases
encode the expected semantics directly and random walks verify incremental vs
full recomputation plus make/unmake restoration.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepblue import fastcore as F  # noqa: E402
from deepblue import zobrist as Z  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def full(position) -> np.uint64:
    bb, occ, _, st = position
    return np.uint64(Z.full_hash(bb, occ, st, Z.PIECE_KEYS, Z.SIDE_KEY, Z.CASTLE_KEYS, Z.EP_FILE_KEYS))


def targeted_ep() -> None:
    cases = [
        (
            "unusable raw EP is ignored",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            True,
        ),
        (
            "legal EP changes identity",
            "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
            "7k/8/8/3pP3/8/8/8/K7 w - - 0 1",
            False,
        ),
        (
            "pinned EP is ignored",
            "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "4r2k/8/8/3pP3/8/8/8/4K3 w - - 0 1",
            True,
        ),
    ]
    for name, a, b, should_equal in cases:
        ha, hb = full(F.from_fen(a)), full(F.from_fen(b))
        if (ha == hb) != should_equal:
            raise AssertionError(f"{name}: {int(ha):016x} vs {int(hb):016x}")
        print(f"pass  {name}")


def random_walks(positions: int, seed: int) -> None:
    rng = random.Random(seed)
    legal = np.zeros((F.MAX_PLY, F.MAX_MOVES), dtype=np.uint32)
    pseudo = np.zeros((F.MAX_PLY, F.MAX_MOVES), dtype=np.uint32)
    undo = np.zeros(F.MAX_PLY * F.UNDO_STRIDE, dtype=np.int64)
    position = F.from_fen(START)
    bb, occ, mail, st = position
    value = full(position)
    checked = 0

    for index in range(positions):
        count = F.generate_legal(bb, occ, mail, st, legal, pseudo, undo, 0)
        if count == 0 or index % 180 == 179:
            position = F.from_fen(START)
            bb, occ, mail, st = position
            value = full(position)
            continue

        move = legal[0, rng.randrange(count)]
        old_castle = int(st[1])
        old_ep_file = int(Z.canonical_ep_file(bb, occ, st))
        before = value
        F.make_move(bb, occ, mail, st, move, undo, 0)
        new_ep_file = int(Z.canonical_ep_file(bb, occ, st))
        value = np.uint64(Z.apply_move(
            value, np.uint64(move), 1 - int(st[0]), old_castle, old_ep_file,
            int(st[1]), new_ep_file, Z.PIECE_KEYS, Z.SIDE_KEY, Z.CASTLE_KEYS, Z.EP_FILE_KEYS,
        ))
        recomputed = full((bb, occ, mail, st))
        if value != recomputed:
            raise AssertionError(f"incremental mismatch at sample {index}")

        F.unmake_move(bb, occ, mail, st, move, undo, 0)
        restored = full((bb, occ, mail, st))
        if restored != before:
            raise AssertionError(f"unmake hash mismatch at sample {index}")
        # Re-apply to continue the walk.
        F.make_move(bb, occ, mail, st, move, undo, 0)
        checked += 1

    print(f"incremental/full + unmake: 0 failures / {checked:,} moves")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--positions", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=20260901)
    a = p.parse_args()
    targeted_ep()
    random_walks(a.positions, a.seed)


if __name__ == "__main__":
    main()
