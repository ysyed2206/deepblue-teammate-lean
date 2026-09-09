"""Does 1000ms/move still reach depths where our techniques fire?

The 60ms tests that wrongly rejected LMP, NMP, LMR and razoring failed because
at 60ms this engine reaches depth 4 and those techniques are depth-gated --
they could not execute. Halving the test time control is only safe if the
depths stay clear of every gate we use:
    LMP <=3, razoring <=3, futility <=3, RFP <=?, NMP >=4, LMR >=3
"""
import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch118 import FastEngine118
FENS = [
 ("italian",  "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
 ("round58",  "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"),
 ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
 ("midgame",  "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"),
 ("endgame",  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
 ("round60",  "r1b1r1k1/pp1n1p1p/4pbp1/6NP/2Pp4/3Q2P1/q2B1PB1/3RR1K1 w - - 0 18"),
]
e = FastEngine118(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
print("%-9s %8s %8s" % ("position", "1000ms", "2000ms"))
d1 = d2 = 0
for name, f in FENS:
    a = e.search(from_fen(f), 1000, 1400)[2]
    b = e.search(from_fen(f), 2000, 2800)[2]
    d1 += a; d2 += b
    print("%-9s %8d %8d" % (name, a, b))
print("\nmean depth: %.1f at 1000ms, %.1f at 2000ms" % (d1/len(FENS), d2/len(FENS)))
print("deepest gate in use is NMP at depth>=4; LMR >=3.")
