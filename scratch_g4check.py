from deepblue.fastcore import from_fen
F = "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"
from deepblue.fastsearch118 import FastEngine118
from deepblue.fastsearch123 import FastEngine123
from deepblue.fastsearch124 import FastEngine124
for label, cls in (("118", FastEngine118), ("123", FastEngine123), ("124", FastEngine124)):
    e = cls(); e.search(from_fen(F), 200, 400)          # warm
    out = []
    for ms in (2000, 3000):
        mv, sc, d, n, el = e.search(from_fen(F), ms, int(ms * 1.4))
        out.append("%4dms -> %s %+5d d%-2d" % (ms, mv, sc, d))
    print("%-4s %s   %s   %s" % (label, out[0], out[1], "BLUNDERS g4" if out[0].split("-> ")[1].startswith("g2g4") else "avoids g4"))
