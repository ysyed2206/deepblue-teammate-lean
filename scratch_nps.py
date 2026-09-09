import time, chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch118 import FastEngine118
from deepblue.fastsearch127 import FastEngine127
FENS = ["r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"]
for label, cls in (("118", FastEngine118), ("127", FastEngine127)):
    e = cls(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
    tot_n = 0; tot_t = 0.0; depths = []
    for f in FENS:
        t = time.time()
        mv, sc, d, n, el = e.search(from_fen(f), 2000, 2800)
        tot_t += time.time() - t; tot_n += n; depths.append(d)
    print("%s  %9d nodes in %.2fs = %,d nps   depths %s".replace(",d","d") %
          (label, tot_n, tot_t, int(tot_n/tot_t), depths))
