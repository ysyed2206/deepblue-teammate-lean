"""How often does an interior node actually have a transposition-table entry?

This decides whether IID / IIR is worth building at all. Both techniques only
do anything at nodes with NO tt move: if the table usually has one, the
ceiling on either technique is near zero no matter how good the idea is.
"""
import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch118 import FastEngine118, NODES, QNODES, TTPROBE, TTHIT

FENS = [
 ("italian",  "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
 ("round58",  "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"),
 ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
 ("midgame",  "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"),
 ("endgame",  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]
e = FastEngine118()
e.search(from_fen(chess.STARTING_FEN), 200, 400)     # JIT
print("%-9s %10s %10s %10s %8s" % ("position", "nodes", "tt probes", "tt hits", "hit%"))
tot_p = tot_h = 0
for name, f in FENS:
    # search() zeroes self.counters at entry (fastsearch118.py:1077), so the
    # counters are ALREADY per-search. Differencing them across searches gave
    # negative probe counts and a 372% hit rate.
    e.search(from_fen(f), 2000, 2800)
    p = int(e.counters[TTPROBE])
    h = int(e.counters[TTHIT])
    tot_p += p; tot_h += h
    print("%-9s %10d %10d %10d %7.1f%%" % (name, int(e.counters[NODES]), p, h,
                                           100.0*h/p if p else 0.0))
print("\noverall tt hit rate: %.1f%%   (miss rate %.1f%% -- the fraction of "
      "nodes where IID/IIR could fire)" % (100.0*tot_h/tot_p, 100.0*(tot_p-tot_h)/tot_p))
