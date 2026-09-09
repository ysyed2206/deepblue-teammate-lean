"""Track [SetUp]/[FEN] starting positions seen across real qualification PGNs.

Working hypothesis (2026-09-05): the competition's per-game starting position
is not move 1 -- every PGN reviewed so far carries a [SetUp "1"]/[FEN ...]
header placing the game 6-8 moves in. Round 25 and round 26's seed FENs are
one legal move apart (round 25 + 7...Nge7 == round 26's own seed), and round
28 shares the same broad structural family (Black c6/e6 knight setup vs
White Nc3/d3), across three different opponents. That is suggestive of a
reused or template-generated seed pool, not confirmed as one -- this tool
exists to keep the evidence honest as more games come in, not to declare
victory on six data points.

Usage: point it at a directory of PGN files; it extracts every [Round] and
[FEN] header pair, and flags any two seeds that are identical or one legal
move apart from each other.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import chess

HEADER_RE = re.compile(r'\[(Round|FEN)\s+"([^"]*)"\]')


def extract_seeds(pgn_paths: list[Path]) -> dict[str, str]:
    """Return {round_number: fen} for every PGN that carries a [SetUp] FEN."""
    seeds: dict[str, str] = {}
    for path in pgn_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        round_no = None
        fen = None
        for match in HEADER_RE.finditer(text):
            key, value = match.groups()
            if key == "Round":
                round_no = value
            elif key == "FEN":
                fen = value
        if round_no is not None and fen is not None:
            seeds[round_no] = fen
    return seeds


def one_move_apart(fen_a: str, fen_b: str) -> str | None:
    """Return the SAN move if playing one legal move from fen_a reaches fen_b
    (comparing board+turn+castling+ep, ignoring move counters), else None."""
    try:
        board_a = chess.Board(fen_a)
        board_b = chess.Board(fen_b)
    except ValueError:
        return None
    target = board_b.fen().split(" ")[:4]
    for move in board_a.legal_moves:
        board_a.push(move)
        if board_a.fen().split(" ")[:4] == target:
            return board_a.san(move) if False else move.uci()
        board_a.pop()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn_dir", type=Path)
    args = ap.parse_args()

    pgn_paths = sorted(args.pgn_dir.glob("*.pgn"))
    seeds = extract_seeds(pgn_paths)
    def sort_key(round_no: str) -> tuple[int, str]:
        return (0, f"{int(round_no):08d}") if round_no.isdigit() else (1, round_no)

    print(f"found {len(seeds)} seed positions across {len(pgn_paths)} PGNs\n")
    for round_no, fen in sorted(seeds.items(), key=lambda kv: sort_key(kv[0])):
        print(f"round {round_no}: {fen}")

    print("\n=== pairwise comparison ===")
    rounds = sorted(seeds.keys(), key=sort_key)
    for i, r1 in enumerate(rounds):
        for r2 in rounds[i + 1:]:
            f1, f2 = seeds[r1], seeds[r2]
            if f1.split(" ")[:4] == f2.split(" ")[:4]:
                print(f"round {r1} == round {r2}: IDENTICAL POSITION")
                continue
            mv = one_move_apart(f1, f2)
            if mv:
                print(f"round {r1} -> round {r2}: ONE MOVE APART ({mv})")
                continue
            mv = one_move_apart(f2, f1)
            if mv:
                print(f"round {r2} -> round {r1}: ONE MOVE APART ({mv})")


if __name__ == "__main__":
    main()
