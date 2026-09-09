"""Speed of 161 (fused eval) vs 150, at FIXED DEPTH.

Fixed depth means both engines search the same tree -- the arithmetic is
verified identical -- so node counts must match exactly and the only
difference is time. That makes this a clean speed measurement, and any
node-count difference would indicate the fusion is NOT behaviour-neutral
after all.
"""
import time, chess
from deepblue.fastcore import from_fen
from deepblue.fastsearch150 import FastEngine150
from deepblue.fastsearch161 import FastEngine161

FENS = [("italian", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("midgame", "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"),
        ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ("round58", "2r3k1/p3rppp/Pp1q1n2/2pp1b2/Q7/2P1P2P/1P1NBPP1/R2R2K1 w - - 1 18")]
DEPTH = 9
HUGE = 3_600_000.0
res = {}
for label, cls in (("150", FastEngine150), ("161", FastEngine161)):
    e = cls(); e.search(from_fen(chess.STARTING_FEN), 200, 400)
    rows = []
    for name, f in FENS:
        t = time.perf_counter()
        mv, sc, d, n, _ = e.search(from_fen(f), HUGE, HUGE, max_depth=DEPTH)
        rows.append((name, mv, n, time.perf_counter() - t))
    res[label] = rows
print("%-9s %-18s %-18s %8s" % ("position", "150 (nodes/time)", "161 (nodes/time)", "speedup"))
tn = t150 = t161 = 0
same = True
for i, (name, _) in enumerate(FENS):
    a, b = res["150"][i], res["161"][i]
    same &= (a[1] == b[1] and a[2] == b[2])
    t150 += a[3]; t161 += b[3]; tn += a[2]
    print("%-9s %8d %8.2fs  %8d %8.2fs  %7.3fx%s"
          % (name, a[2], a[3], b[2], b[3], a[3]/b[3], "" if a[2]==b[2] else "  NODES DIFFER"))
print()
print("identical trees: %s" % same)
print("total: 150 %.2fs -> 161 %.2fs   speedup %.3fx  (%.1f%% faster)"
      % (t150, t161, t150/t161, (t150/t161 - 1) * 100))
