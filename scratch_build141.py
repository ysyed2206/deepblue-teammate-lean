"""fastsearch141 = 140 + bishop/rook weights re-proportioned.

The classic attack-unit scheme used by engines WITHOUT Stockfish's safe-check
machinery weights minor pieces 2, rooks 3, queens 5 -- rook above minors,
queen highest. Modern Stockfish inverts that (N81 B52 R44 Q10) because the
queen's and rook's danger moved into separate check terms we never
implemented. Scaled so the knight keeps its current 81:

    knight  81  (unchanged)
    bishop  52 -> 81   equal to the knight, as the classic scheme has it
    rook    44 -> 121  above the minors, where it belongs
    queen   10 -> 90   (already changed and measured separately in 140)

Two constants, bundled deliberately: each is worth only a few Elo in the rare
positions where it fires, so testing them one at a time would burn hours to
produce two unresolvable results. Their signs are fixed by the published
scheme, which is the condition that makes bundling legitimate here.
"""
from pathlib import Path
p = Path("deepblue/eval_terms.py"); s = p.read_text(encoding="utf-8")
for old, new in (("KING_ZONE_ATTACK_WEIGHT_B = 52", "KING_ZONE_ATTACK_WEIGHT_B = 81"),
                 ("KING_ZONE_ATTACK_WEIGHT_R = 44", "KING_ZONE_ATTACK_WEIGHT_R = 121")):
    assert s.count(old) == 1, old
    s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
src = Path("deepblue/fastsearch140.py"); dst = Path("deepblue/fastsearch141.py")
t = src.read_text(encoding="utf-8")
dst.write_text(t.replace("FastEngine140", "FastEngine141").replace("S1-search140", "S1-search141"),
               encoding="utf-8")
print("wrote fastsearch141 = 140 + bishop 52->81, rook 44->121")
