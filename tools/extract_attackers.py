"""Per-piece-type king-zone attacker counts, White-relative.

Enough to fit the four KING_ZONE_ATTACK_WEIGHT constants directly against
Stockfish scores instead of choosing them by analogy to another engine.
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import chess
import numpy as np

TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
NAMES = ["wN","wB","wR","wQ","bN","bB","bR","bQ"]


def side(b, us):
    ksq = b.king(us)
    if ksq is None:
        return [0.0] * 4
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[ksq]) | chess.SquareSet(chess.BB_SQUARES[ksq])
    out = []
    for pt in TYPES:
        n = 0
        for sq in b.pieces(pt, not us):
            if chess.SquareSet(b.attacks_mask(sq)) & zone:
                n += 1
        out.append(float(n))
    return out


def one(line):
    cp, fen = line.rstrip("\n").split("\t", 1)
    try:
        b = chess.Board(fen)
    except ValueError:
        return None
    w, k = side(b, chess.WHITE), side(b, chess.BLACK)
    # Both sides kept separately: the penalty is a per-side nonlinear function
    # of that king's attackers, so differencing the COUNTS first would fit the
    # wrong thing. Columns 0-3 = attackers on White's king, 4-7 = on Black's.
    return w + k


def main():
    src, out, limit, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    lines = open(src, encoding="utf-8").readlines()[:limit]
    with mp.Pool(workers) as pool:
        rows = pool.map(one, lines, chunksize=400)
    keep = [i for i, r in enumerate(rows) if r is not None]
    A = np.array([rows[i] for i in keep], dtype=np.float64)
    np.savez_compressed(out, A=A, keep=np.array(keep), names=np.array(NAMES))
    print("wrote %s %s" % (out, A.shape))


if __name__ == "__main__":
    mp.freeze_support()
    main()
