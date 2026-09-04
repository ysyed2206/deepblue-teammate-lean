"""Simulate one side's clock across a full game and report time discipline.

The referee measures wall time around the whole call and a flag is an instant
loss, so what matters is not the average but the worst move and whether the
clock ever runs out. This drives real self-play so the positions are real.
"""

from __future__ import annotations

import argparse
import sys
import time

import chess

sys.path.insert(0, ".")
import agent  # noqa: E402
from deepblue.time_manager import allocate  # noqa: E402

BASE_MS = 120_000
INCREMENT_MS = 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plies", type=int, default=120)
    parser.add_argument("--base-ms", type=int, default=BASE_MS)
    arguments = parser.parse_args()

    board = chess.Board()
    clocks = {chess.WHITE: float(arguments.base_ms), chess.BLACK: float(arguments.base_ms)}
    played = {chess.WHITE: 0, chess.BLACK: 0}
    spent: list[float] = []
    flagged = None

    for _ in range(arguments.plies):
        if board.is_game_over(claim_draw=True):
            break
        mover = board.turn
        agent._moves_played = played[mover]
        started = time.monotonic()
        uci = agent.get_move(board.fen(), int(clocks[mover]))
        elapsed = (time.monotonic() - started) * 1000.0
        clocks[mover] -= elapsed
        spent.append(elapsed)
        if clocks[mover] < 0:
            flagged = (mover, len(spent))
            break
        clocks[mover] += INCREMENT_MS
        played[mover] += 1
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"ILLEGAL MOVE {uci} at {board.fen()}", file=sys.stderr)
            raise SystemExit(1)
        board.push(move)

    print(f"plies played      : {len(spent)}")
    print(f"per-move ms       : min {min(spent):.0f}  mean {sum(spent) / len(spent):.0f}  max {max(spent):.0f}")
    print(f"clock left  white : {clocks[chess.WHITE]:,.0f} ms")
    print(f"clock left  black : {clocks[chess.BLACK]:,.0f} ms")
    print(f"flagged           : {flagged if flagged else 'no'}")
    print(f"result            : {board.result(claim_draw=True)}  ({board.fen()})")


if __name__ == "__main__":
    main()
