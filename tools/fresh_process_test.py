"""Torture the submission the way the platform will: from a fresh process, from
the packaged layout, one process per game.

A build that survives a thousand calls inside one long-lived development
process is not evidence of anything. The platform starts a new process for
every game, imports agent.py from the unpacked zip, and gives that import 60
seconds. This test reproduces exactly that, repeatedly.

It deliberately builds the zip and runs out of the *extracted* zip, not the
repository, because the packager globs only top-level *.py plus named includes
- a submission that omits the engine package imports fine in the repo and dies
on the platform.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent

# Positions chosen for the edges that break move generation and search.
PROBE_POSITIONS: list[tuple[str, str]] = [
    ("startpos", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("black to move", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
    ("promotion available", "8/P6k/8/8/8/8/6K1/8 w - - 0 1"),
    ("underpromotion matters", "8/1P5k/8/8/8/8/6K1/7q w - - 0 1"),
    ("en passant available", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    ("castling both sides", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),
    ("in check, one legal move", "7k/8/8/8/8/8/5rr1/K7 w - - 0 1"),
    ("stalemate adjacent", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"),
    ("bare kings", "8/8/4k3/8/8/4K3/8/8 w - - 0 1"),
    ("no castling rights", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1"),
    ("deep endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("many queens", "q6k/8/8/8/8/8/8/Q5K1 w - - 0 1"),
    ("high halfmove clock", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 99 60"),
]

CHILD = r'''
import json, sys, time, importlib
sys.path.insert(0, sys.argv[1])
started = time.perf_counter()
agent = importlib.import_module("agent")
init_ms = (time.perf_counter() - started) * 1000.0
results = []
for name, fen, clock in json.loads(sys.argv[2]):
    t = time.perf_counter()
    move = agent.get_move(fen, clock)
    results.append([name, fen, move, (time.perf_counter() - t) * 1000.0])
print("@@RESULT@@" + json.dumps({"init_ms": init_ms, "moves": results}))
'''


def build_zip(destination: Path) -> Path:
    archive = destination / "submission.zip"
    subprocess.run(
        # Deliberately the BARE command, with no --include, so that a
        # regression in the default include list fails this test.
        [sys.executable, "-m", "harness.package", "--out", str(archive)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    extracted = destination / "unpacked"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(extracted)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clock-ms", type=int, default=3000)
    arguments = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="deepblue-fresh-"))
    try:
        extracted = build_zip(workspace)
        shipped = sorted(p.relative_to(extracted).as_posix() for p in extracted.rglob("*") if p.is_file())
        unzipped_bytes = sum(p.stat().st_size for p in extracted.rglob("*") if p.is_file())
        print(f"packaged {len(shipped)} files, {unzipped_bytes:,} bytes unzipped")
        for name in shipped:
            print(f"    {name}")
        if "agent.py" not in shipped:
            print("FAIL: agent.py is not at the zip root")
            raise SystemExit(1)
        print()

        probes = json.dumps([[n, f, arguments.clock_ms] for n, f in PROBE_POSITIONS])
        init_times: list[float] = []
        move_times: list[float] = []
        failures = 0

        for round_index in range(arguments.rounds):
            completed = subprocess.run(
                [sys.executable, "-c", CHILD, str(extracted), probes],
                capture_output=True,
                text=True,
                timeout=300,
            )
            marker = [ln for ln in completed.stdout.splitlines() if ln.startswith("@@RESULT@@")]
            if completed.returncode != 0 or not marker:
                print(f"round {round_index + 1}: PROCESS FAILED rc={completed.returncode}")
                print(completed.stderr[-2000:])
                failures += 1
                continue
            payload = json.loads(marker[0][len("@@RESULT@@"):])
            init_times.append(payload["init_ms"])
            round_bad = 0
            for name, fen, move, elapsed in payload["moves"]:
                move_times.append(elapsed)
                board = chess.Board(fen)
                legal = [m.uci() for m in board.legal_moves]
                if not legal:
                    continue
                if move not in legal:
                    print(f"  ILLEGAL: {name!r} fen={fen} returned {move!r}")
                    round_bad += 1
                if elapsed > arguments.clock_ms:
                    print(f"  OVER CLOCK: {name!r} took {elapsed:.0f}ms of {arguments.clock_ms}ms")
                    round_bad += 1
            failures += round_bad
            print(f"round {round_index + 1}/{arguments.rounds}: import {payload['init_ms']:6.0f} ms, "
                  f"{len(payload['moves'])} probes, {round_bad} problems")

        print()
        print(f"cold import ms    : min {min(init_times):.0f}  max {max(init_times):.0f}  "
              f"(budget 60,000)")
        print(f"per-move ms       : min {min(move_times):.0f}  max {max(move_times):.0f}  "
              f"(clock {arguments.clock_ms})")
        print(f"total problems    : {failures}")
        raise SystemExit(1 if failures else 0)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
