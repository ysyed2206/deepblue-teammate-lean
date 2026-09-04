"""Differential gate for Deep Blue Static Exchange Evaluation.

The fast SEE is compared with a deliberately slow python-chess oracle which
plays only legal captures back onto the exchange square and lets either side
decline the exchange.  Patch 06 uses SEE for ordering only, so the most
important metric is the sign (winning/non-losing vs losing), not exact cp.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import MAX_MOVES, from_fen, generate_pseudo_legal, move_to_uci
from deepblue.see import see_value

VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def immediate_gain(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        captured = chess.PAWN
    else:
        piece = board.piece_at(move.to_square)
        captured = piece.piece_type if piece else None
    gain = VALUE[captured] if captured is not None else 0
    if move.promotion is not None:
        gain += VALUE[move.promotion] - VALUE[chess.PAWN]
    return gain


def exchange_oracle(board: chess.Board, target: chess.Square) -> int:
    """Best material the side to move can force by continuing captures on target."""
    best = 0
    candidates = [
        move for move in board.legal_moves
        if move.to_square == target and board.is_capture(move)
    ]
    for move in candidates:
        gain = immediate_gain(board, move)
        board.push(move)
        score = gain - exchange_oracle(board, target)
        board.pop()
        if score > best:
            best = score
    return best


def oracle_for_move(board: chess.Board, move: chess.Move) -> int:
    gain = immediate_gain(board, move)
    target = move.to_square
    board.push(move)
    result = gain - exchange_oracle(board, target)
    board.pop()
    return result


def packed_move_for(board: chess.Board, move: chess.Move, buf: np.ndarray) -> int:
    position = from_fen(board.fen())
    bb, occ, mail, st = position
    count = generate_pseudo_legal(bb, occ, mail, st, buf)
    want = move.uci()
    for i in range(count):
        packed = int(buf[i])
        if move_to_uci(packed) == want:
            return packed
    raise RuntimeError(f"generated legal move {want} not found in Deep Blue pseudo list")


REGRESSION_CASES = [
    # A recapturing pawn promotes on the exchange square.  SEE must include
    # both the capture and promotion swing; this was the Patch 06 sign bug.
    ("B3k3/1Pn3q1/8/3P4/1rp1p1b1/6R1/5K2/8 b - - 1 65", "c7a8", -800),
    # Colour-flipped mirror of the same motif.
    ("8/5k2/6r1/1RP1P1B1/3p4/8/1pN3Q1/b3K3 w - - 1 65", "c2a1", -800),
]


def run_regressions(buf: np.ndarray) -> None:
    for fen, uci, expected in REGRESSION_CASES:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise AssertionError(f"bad SEE regression: {uci} is not legal in {fen}")
        packed = packed_move_for(board, move, buf)
        bb, occ, mail, st = from_fen(fen)
        fast = int(see_value(bb, occ, mail, st, np.uint32(packed)))
        slow = int(oracle_for_move(board, move))
        if slow != expected or fast != slow:
            raise AssertionError(
                f"SEE regression failed: {uci} fast={fast:+d} oracle={slow:+d} "
                f"expected={expected:+d}  {fen}"
            )
    print(f"SEE regression motifs: {len(REGRESSION_CASES)}/{len(REGRESSION_CASES)} passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=3000)
    ap.add_argument("--max-captures", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    board = chess.Board()
    buf = np.zeros(MAX_MOVES, dtype=np.uint32)
    run_regressions(buf)
    tested = exact_bad = sign_bad = 0
    examples: list[str] = []

    for _ in range(args.positions):
        # Periodically restart; otherwise take a short random legal walk to get
        # a broad mix of opening/middlegame/endgame structures.
        if board.is_game_over(claim_draw=True) or rng.random() < 0.08:
            board = chess.Board()
        plies = rng.randint(1, 10)
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                board = chess.Board()
                break
            moves = list(board.legal_moves)
            board.push(rng.choice(moves))

        captures = [m for m in board.legal_moves if board.is_capture(m) and m.promotion is None]
        rng.shuffle(captures)
        for move in captures[:4]:
            fen = board.fen()
            packed = packed_move_for(board, move, buf)
            bb, occ, mail, st = from_fen(fen)
            fast = int(see_value(bb, occ, mail, st, np.uint32(packed)))
            slow = int(oracle_for_move(board, move))
            tested += 1
            if fast != slow:
                exact_bad += 1
            if (fast >= 0) != (slow >= 0):
                sign_bad += 1
                if len(examples) < 12:
                    examples.append(f"{move.uci()} fast={fast:+d} oracle={slow:+d}  {fen}")
            if tested >= args.max_captures:
                break
        if tested >= args.max_captures:
            break

    print(f"seed {args.seed}, sampled positions {args.positions}")
    print(f"SEE captures tested: {tested:,}")
    print(f"exact-value mismatches: {exact_bad:,}")
    print(f"sign mismatches:        {sign_bad:,}")
    if examples:
        print("first sign mismatches:")
        for row in examples:
            print("  " + row)
    if tested == 0:
        raise SystemExit("no captures sampled")
    if sign_bad:
        raise SystemExit(1)
    print("PASS: fast SEE agrees with the legal oracle on every tested capture sign")


if __name__ == "__main__":
    main()
