import time
from deepblue.fastcore import from_fen
from deepblue.fastsearch118 import FastEngine118
pos = [("18.g4",   "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"),
       ("19.Nc4",  "2r3k1/p4ppp/Pp1q1n2/2pp1b2/Q5P1/2P1r2P/1P1NBP2/R2R2K1 w - - 0 19"),
       ("21.Nd6",  "2r3k1/p4ppp/Pp3n2/2pp1b2/Q1N2qP1/2P3rP/1P2BP2/R2R1K2 w - - 4 21"),
       ("25.Rxd5", "2r3k1/p4ppp/P2q1n2/1ppp1P2/Q4P2/2P2B2/1P5r/R2R1K2 w - - 0 25")]
e = FastEngine118()
e.search(from_fen(pos[0][1]), 200, 400)          # warm the JIT
print("%-8s %-22s %-22s %-22s" % ("played", "2s (game speed)", "8s", "20s"))
for name, f in pos:
    row = []
    for ms in (2000, 8000, 20000):
        mv, sc, d, n, el = e.search(from_fen(f), ms, int(ms * 1.4))
        row.append("%s %+5d d%-2d" % (mv, sc, d))
    print("%-8s %-22s %-22s %-22s" % (name, row[0], row[1], row[2]))
