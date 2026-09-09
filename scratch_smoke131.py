import time, chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch118 import FastEngine118
from deepblue.fastsearch131 import FastEngine131
FENS = [("italian","r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("round58","2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"),
        ("kiwipete","r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ("midgame","r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"),
        ("endgame","8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ("round60","r1b1r1k1/pp1n1p1p/4pbp1/6NP/2Pp4/3Q2P1/q2B1PB1/3RR1K1 w - - 0 18")]
res = {}
for label, cls in (("118", FastEngine118), ("131", FastEngine131)):
    e = cls(); t = time.time(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
    print("%s compile %.0fs" % (label, time.time()-t))
    res[label] = [e.search(from_fen(f), 2000, 2800) for _, f in FENS]
print("\n%-9s %-22s %-22s" % ("pos", "118  move d nodes", "131  move d nodes"))
d1 = d2 = n1 = n2 = 0
for i, (name, _) in enumerate(FENS):
    a, b = res["118"][i], res["131"][i]
    d1 += a[2]; d2 += b[2]; n1 += a[3]; n2 += b[3]
    print("%-9s %-4s d%-2d %8d      %-4s d%-2d %8d" % (name, a[0], a[2], a[3], b[0], b[2], b[3]))
print("\ntotal depth  118=%d  131=%d  (%+d)" % (d1, d2, d2-d1))
print("total nodes  118=%d  131=%d  (%+.1f%%)" % (n1, n2, 100.0*(n2-n1)/n1))
