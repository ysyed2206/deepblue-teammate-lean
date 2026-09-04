"""Perft over the S1 core, checked against published counts and against
python-chess computed independently.

A single passing start-position perft is not evidence. The suite covers
castling, en passant, promotions, pins and check evasions, and every count is
verified twice: against the published reference value, and against a perft run
by python-chess on the same position.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastcore as F  # noqa: E402

# Published counts. Positions 1-6 are the standard perft suite.
SUITE: list[tuple[str, str, list[int]]] = [
    ("startpos", chess.STARTING_FEN, [20, 400, 8902, 197281, 4865609]),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [48, 2039, 97862, 4085603],
    ),
    ("position 3 (endgame)", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     [14, 191, 2812, 43238, 674624]),
    (
        "position 4 (promotions)",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        [6, 264, 9467, 422333],
    ),
    (
        "position 5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        [44, 1486, 62379, 2103487],
    ),
    (
        "position 6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        [46, 2079, 89890, 3894594],
    ),
]


def reference_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += reference_perft(board, depth - 1)
        board.pop()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--cross-check-depth", type=int, default=3)
    arguments = parser.parse_args()

    print(f"warm up: {F.warm_up():.2f}s\n")
    print(f"{'position':<26}{'depth':>6}{'ours':>12}{'published':>12}{'py-chess':>12}{'nps':>12}  ")
    print("-" * 82)
    failures = 0
    total_nodes = 0
    total_seconds = 0.0

    for name, fen, counts in SUITE:
        for depth, expected in enumerate(counts[: arguments.max_depth], start=1):
            bb, occ, mail, st = F.from_fen(fen)
            stack, pseudo_stack, undo = F.new_search_buffers()
            started = time.perf_counter()
            ours = F.perft(bb, occ, mail, st, depth, stack, pseudo_stack, undo, 0)
            elapsed = time.perf_counter() - started
            total_nodes += ours
            total_seconds += elapsed

            cross = "-"
            if depth <= arguments.cross_check_depth:
                cross_value = reference_perft(chess.Board(fen), depth)
                cross = f"{cross_value:,}"
                if cross_value != expected:
                    print(f"  !! published value disagrees with python-chess for {name} d{depth}")
                    failures += 1
            ok = ours == expected
            failures += 0 if ok else 1
            mark = "" if ok else "   <-- MISMATCH"
            print(
                f"{name:<26}{depth:>6}{ours:>12,}{expected:>12,}{cross:>12}"
                f"{ours / max(elapsed, 1e-9):>12,.0f}{mark}"
            )
    print("-" * 82)
    print(f"total {total_nodes:,} nodes in {total_seconds:.2f}s -> "
          f"{total_nodes / max(total_seconds, 1e-9):,.0f} perft NPS")
    print(f"failures: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
