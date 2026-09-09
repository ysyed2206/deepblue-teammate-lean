from deepblue.fastcore import from_fen
from deepblue.eval_terms import (king_safety_white_relative,
                                 king_safety_graded_white_relative, game_phase)
from deepblue.fastsearch118 import PHASE_TABLE
cases = [
  ("before 17.h3   f2 g2 h2", "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P3/1P1NBPPP/R2R2K1 w - - 0 17"),
  ("after  17.h3   f2 g2 h3", "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18"),
  ("after  18.g4   f2 g4 h3", "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q5P1/2P1P2P/1P1NBP2/R2R2K1 b - - 0 18"),
  ("symmetric start        ", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
  ("both castled, intact   ", "r4rk1/pppppppp/8/8/8/8/PPPPPPPP/R4RK1 w - - 0 1"),
]
print("%-26s %8s %8s" % ("position", "flat", "graded"))
for name, f in cases:
    bb, occ, mail, st = from_fen(f)
    ph = game_phase(bb, PHASE_TABLE, 24)
    a = int(king_safety_white_relative(bb, ph, 24))
    b = int(king_safety_graded_white_relative(bb, ph, 24))
    print("%-26s %+8d %+8d" % (name, a, b))
