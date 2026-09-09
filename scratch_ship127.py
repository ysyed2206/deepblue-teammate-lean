import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch127 import FastEngine127
e = FastEngine127(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
F = "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"
for ms in (2000, 3000, 5000):
    mv, sc, d, n, el = e.search(from_fen(F), ms, int(ms*1.4))
    print("127 %5dms -> %s %+5d d%-2d %s" % (ms, mv, sc, d, "PLAYS g4" if mv=="g2g4" else "avoids g4"))
