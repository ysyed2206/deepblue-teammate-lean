"""Colour-symmetry check: eval(P) must equal eval(mirror(P)).

board.mirror() flips the board vertically AND swaps colours, so the mirrored
position is the same position seen from the other side. Our evaluate() is
side-to-move relative, so the two MUST score identically. Any difference is a
colour bias baked into the evaluation -- which would show up in real play as
scoring worse with one colour, exactly the pattern the competition record
hints at (51.9% as White, 41.7% as Black).
"""
import chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch118 import evaluate, MG_TABLE, EG_TABLE, PHASE_TABLE

FENS = [
 "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
 "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18",
 "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
 "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11",
 "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
 "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 7",
 "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPP1BPPP/R2Q1RK1 w - - 0 9",
 "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
 "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
 "8/8/8/3k4/8/3K4/6P1/8 w - - 0 1",
]
print("%-4s %8s %8s %6s" % ("#", "eval", "mirrored", "diff"))
worst = 0
for i, f in enumerate(FENS, 1):
    b = chess.Board(f)
    a = int(evaluate(*[from_fen(f)[j] for j in (0,)], from_fen(f)[3], MG_TABLE, EG_TABLE, PHASE_TABLE))
    mf = b.mirror().fen()
    c = int(evaluate(from_fen(mf)[0], from_fen(mf)[3], MG_TABLE, EG_TABLE, PHASE_TABLE))
    d = a - c
    worst = max(worst, abs(d))
    print("%-4d %8d %8d %6d%s" % (i, a, c, d, "   <-- ASYMMETRIC" if d else ""))
print("\nlargest asymmetry: %d cp" % worst)
print("PASS - evaluation is colour-symmetric" if worst == 0 else
      "FAIL - the evaluation scores the same position differently by colour")
