import time, chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch132 import FastEngine132
e = FastEngine132()
t = time.time(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
print("compile %.0fs" % (time.time()-t))
for name, f in [("italian","r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
                ("round58","2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"),
                ("kiwipete","r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
                ("midgame","r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"),
                ("endgame","8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
                ("round60","r1b1r1k1/pp1n1p1p/4pbp1/6NP/2Pp4/3Q2P1/q2B1PB1/3RR1K1 w - - 0 18")]:
    mv, sc, d, n, ms = e.search(from_fen(f), 2000, 2800)
    print("%-9s %-5s d%-2d %8d nodes  %+5d" % (name, mv, d, n, sc))
nz = int((e.corr_hist != 0).sum())
print("\ncorrection-history entries learned: %d of %d" % (nz, e.corr_hist.size))
print("range: %d .. %d" % (int(e.corr_hist.min()), int(e.corr_hist.max())))
