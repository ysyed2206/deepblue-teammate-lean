from deepblue.fastcore import from_fen
F = "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"
from deepblue.fastsearch125 import FastEngine125
from deepblue.fastsearch126 import FastEngine126
for label, cls in (("125 (11/27)", FastEngine125), ("126 (24/52)", FastEngine126)):
    e = cls(); e.search(from_fen(F), 200, 400)
    out = []
    for ms in (2000, 3000, 5000):
        mv, sc, d, n, el = e.search(from_fen(F), ms, int(ms * 1.4))
        out.append("%s %+5d d%-2d" % (mv, sc, d))
    print("%-12s  2s: %-16s 3s: %-16s 5s: %-16s" % (label, out[0], out[1], out[2]))
