import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch131 import FastEngine131, NODES, QNODES, EVAL
from deepblue.fastsearch132 import FastEngine132
FENS = [("round58","2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"),
        ("midgame","r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"),
        ("round60","r1b1r1k1/pp1n1p1p/4pbp1/6NP/2Pp4/3Q2P1/q2B1PB1/3RR1K1 w - - 0 18")]
for label, cls in (("131", FastEngine131), ("132", FastEngine132)):
    e = cls(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
    print("--- %s ---" % label)
    for name, f in FENS:
        mv, sc, d, n, ms = e.search(from_fen(f), 2000, 2800)
        q = int(e.counters[QNODES]); tot = int(e.counters[NODES])
        print("  %-8s d%-2d %8d nodes %7.0fms %8.0f nps  q=%4.1f%%"
              % (name, d, n, ms, n/ms*1000, 100.0*q/tot))
