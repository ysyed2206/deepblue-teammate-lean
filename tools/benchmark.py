"""Measure the engine on a fixed position suite.

Three numbers are reported separately and must never be quoted for each other:

    movegen NPS   positions per second of legal move generation alone
    perft NPS     nodes per second enumerating the tree, no evaluation
    search NPS    nodes per second of the real search, including evaluation,
                  ordering, transposition probes and quiescence

Only the last one bounds playing strength. Every measurement is repeated and
the median is reported, because one lucky run is not a measurement.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.reference import Engine  # noqa: E402

SUITE: list[tuple[str, str]] = [
    ("opening", chess.STARTING_FEN),
    ("open middlegame", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("closed middlegame", "r1bq1rk1/pp2nppp/2n1p3/2ppP3/3P4/P1PB1N2/2P2PPP/R1BQK2R w KQ - 0 1"),
    ("tactical", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]


def movegen_rate(board: chess.Board, seconds: float) -> float:
    """Positions per second, divided by ACTUAL elapsed time.

    The loop runs in batches so the clock is not read every iteration, which
    means it always overruns its nominal deadline slightly. Dividing by the
    requested duration rather than the measured one overstated the rate.
    """
    started = time.perf_counter()
    deadline = started + seconds
    count = 0
    while time.perf_counter() < deadline:
        for _ in range(200):
            list(board.legal_moves)
        count += 200
    return count / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movetime-ms", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()

    print(f"{'position':<20}{'movegen/s':>12}{'search nps':>12}{'depth':>7}{'nodes':>10}")
    print("-" * 61)
    all_nps: list[float] = []
    for name, fen in SUITE:
        board = chess.Board(fen)
        generation = movegen_rate(board, 0.25)
        samples: list[tuple[float, int, int]] = []
        for _ in range(arguments.repeats):
            engine = Engine()
            engine.record_game_position(board)
            result = engine.search(
                chess.Board(fen), arguments.movetime_ms, arguments.movetime_ms * 1.2
            )
            samples.append(
                (result.nodes / max(result.elapsed_ms, 1.0) * 1000.0, result.depth, result.nodes)
            )
        nps = statistics.median(s[0] for s in samples)
        depth = statistics.median(s[1] for s in samples)
        nodes = int(statistics.median(s[2] for s in samples))
        all_nps.append(nps)
        print(f"{name:<20}{generation:>12,.0f}{nps:>12,.0f}{depth:>7.0f}{nodes:>10,}")
    print("-" * 61)
    print(f"{'median search NPS':<20}{'':>12}{statistics.median(all_nps):>12,.0f}")


if __name__ == "__main__":
    main()
