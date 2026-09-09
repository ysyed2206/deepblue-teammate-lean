"""Does ageing actually evict stale entries, and does it change play?

Plays a sequence of positions through ONE engine (as a real game does) and
reports how many table slots carry the current generation afterwards. With no
ageing, old deep entries accumulate and cannot be evicted.
"""
import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch152 import FastEngine152
import numpy as np
FENS = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11",
        "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"]
e = FastEngine152()
e.search(from_fen(FENS[0]), 200, 400)
for i, f in enumerate(FENS):
    mv, sc, d, n, _ = e.search(from_fen(f), 2000, 2800)
    used = int((e.tt_key != 0).sum())
    current = int((e.tt_age == e.tt_generation).sum())
    print("move %d  gen=%3d  best %-6s d%-2d   slots used %8d   current-gen %8d (%.0f%%)"
          % (i + 1, int(e.tt_generation), mv, d, used, current,
             100.0 * current / used if used else 0))
