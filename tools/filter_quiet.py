"""Keep only quiet positions from the Stockfish sample.

Texel tuning fits a STATIC evaluation. On a tactical position the deep
Stockfish score encodes a combination our eval cannot represent at any
setting of its constants, so fitting to it drags every constant toward
absorbing tactics and makes the quiet positions worse. Standard practice
is to tune on quiet positions only; this applies that filter.

Quiet here means: not in check, no check available, and no capture whose
victim outvalues its attacker (a cheap SEE proxy).
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 99}


def quiet(line: str):
    cp, fen = line.rstrip("\n").split("\t", 1)
    try:
        b = chess.Board(fen)
    except ValueError:
        return None
    if b.is_check():
        return None
    for mv in b.legal_moves:
        if b.gives_check(mv):
            return None
        if b.is_capture(mv):
            if b.is_en_passant(mv):
                continue
            victim = b.piece_at(mv.to_square)
            attacker = b.piece_at(mv.from_square)
            if victim and attacker and VAL[victim.piece_type] > VAL[attacker.piece_type]:
                return None
    return line


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    lines = open(src, encoding="utf-8").readlines()
    with mp.Pool(6) as pool:
        kept = [r for r in pool.map(quiet, lines, chunksize=500) if r]
    with open(dst, "w", encoding="utf-8") as fh:
        fh.writelines(kept)
    print("quiet %d of %d (%.1f%%) -> %s"
          % (len(kept), len(lines), 100 * len(kept) / max(1, len(lines)), dst))


if __name__ == "__main__":
    mp.freeze_support()
    main()
