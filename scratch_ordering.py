"""First-move cutoff rate: the standard measure of move-ordering quality.

A ratio, so CPU contention cannot bias it. Strong engines report 90%+.
"""
import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch199 import FastEngine199, CUTOFFS, FIRST_CUTOFFS, NODES, QNODES
FENS = [("italian",  "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ("midgame",  "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"),
        ("r72 end",  "8/8/4kp2/3p4/p6P/P5K1/8/8 w - - 0 40"),
        ("endgame",  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")]
e = FastEngine199(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
tot_c = tot_f = 0
print("%-10s %10s %10s %8s" % ("position", "cutoffs", "on 1st", "rate"))
for name, f in FENS:
    e.search(from_fen(f), 3000, 4200)
    c, fc = int(e.counters[CUTOFFS]), int(e.counters[FIRST_CUTOFFS])
    tot_c += c; tot_f += fc
    print("%-10s %10d %10d %7.1f%%" % (name, c, fc, 100.0*fc/c if c else 0))
print("\noverall first-move cutoff rate: %.1f%%   (strong engines: 90%%+)" % (100.0*tot_f/tot_c))
