"""Differential move generation: S1 against python-chess, over a large random
corpus deliberately weighted toward the positions that break move generators.

python-chess is the oracle. Both sides are reduced to a canonical set of UCI
strings and required to be exactly equal. On any mismatch the FEN, both move
sets, the missing and extra moves and the seed are printed so the case can be
reproduced immediately and added to the regression file.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastcore as F  # noqa: E402

# Seeds for positions that specifically stress the hard cases.
STRESS_FENS = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3",          # en passant exposing check
    "8/8/8/8/k1p4R/8/3P4/3K4 b - - 0 1",            # pinned pawn, ep candidate
    "5k2/8/8/8/8/8/4P3/4K2R w K - 0 1",             # castling with pawn
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",             # both castles available
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "8/P1P5/8/8/8/8/5p1p/4K2k w - - 0 1",           # promotions both sides
    "8/8/8/8/8/8/6k1/4K2R w K - 0 1",
    "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",            # ep pin along rank
    "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",
    "8/5k2/8/2Pp4/2B5/1K6/8/8 w - d6 0 1",
]


def canonical(board: chess.Board) -> set[str]:
    return {move.uci() for move in board.legal_moves}


def ours(fen: str) -> set[str]:
    bb, occ, mail, st = F.from_fen(fen)
    return {F.move_to_uci(move) for move in F.legal_moves(bb, occ, mail, st)}


def check(fen: str, mismatches: list[tuple[str, set[str], set[str]]]) -> bool:
    board = chess.Board(fen)
    theirs = canonical(board)
    mine = ours(fen)
    if mine != theirs:
        mismatches.append((fen, theirs, mine))
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260831)
    arguments = parser.parse_args()

    F.warm_up()
    rng = random.Random(arguments.seed)
    mismatches: list[tuple[str, set[str], set[str]]] = []
    checked = 0

    for fen in STRESS_FENS:
        check(fen, mismatches)
        checked += 1

    # Random walks from the start position and from each stress position, so
    # the corpus reaches sparse endings, promotion races and broken castling.
    origins = [chess.STARTING_FEN] + STRESS_FENS
    while checked < arguments.positions:
        board = chess.Board(rng.choice(origins))
        for _ in range(rng.randrange(0, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            # Bias toward captures and promotions so the walk reaches thin,
            # promotion-heavy endings rather than wandering in the middlegame.
            interesting = [m for m in moves if board.is_capture(m) or m.promotion]
            pool = interesting if interesting and rng.random() < 0.45 else moves
            board.push(rng.choice(pool))
            if not check(board.fen(), mismatches):
                break
            checked += 1
            if checked >= arguments.positions:
                break

    print(f"positions checked : {checked:,}")
    print(f"mismatches        : {len(mismatches)}")
    for fen, theirs, mine in mismatches[:5]:
        print(f"\n  FEN      {fen}")
        print(f"  missing  {sorted(theirs - mine)}")
        print(f"  extra    {sorted(mine - theirs)}")
        print(f"  seed     {arguments.seed}")
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
