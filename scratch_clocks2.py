"""Re-do the phase comparison EXCLUDING near-instant moves.

The 1.50s late-third figure lumps together three different things: the
predictive stop, single-legal-move returns (0ms by design), and mate-score
early exits. Only the first is a defect. Excluding spends under 0.25s
isolates moves where the engine actually chose to think and then stopped.
"""
import glob, statistics, chess, chess.pgn
first, last, zeros_first, zeros_last, n = [], [], 0, 0, 0
for path in sorted(glob.glob(r"C:/Users/uniqu/Downloads/aichessathon-round-*.pgn")):
    g = chess.pgn.read_game(open(path))
    if g is None: continue
    us = chess.WHITE if g.headers.get("White","").lower().startswith("yumo") else chess.BLACK
    node, prev, sp = g, None, []
    while node.variations:
        node = node.variation(0)
        if node.parent.board().turn == us:
            c = node.clock()
            if c is not None:
                if prev is not None: sp.append(prev + 0.5 - c)
                prev = c
    if len(sp) < 12: continue
    n += 1
    t = len(sp)//3
    f, l = sp[:t], sp[-t:]
    zeros_first += sum(1 for x in f if x < 0.25); zeros_last += sum(1 for x in l if x < 0.25)
    f = [x for x in f if x >= 0.25]; l = [x for x in l if x >= 0.25]
    if f: first.append(statistics.mean(f))
    if l: last.append(statistics.mean(l))
print("games: %d" % n)
print("\nincluding every move:")
print("  (previous result: first third 2.87s, last third 1.50s)")
print("\nexcluding moves under 0.25s (forced / mate-score exits):")
print("  first third: %.2f s   (%d instant moves excluded)" % (statistics.mean(first), zeros_first))
print("  last  third: %.2f s   (%d instant moves excluded)" % (statistics.mean(last), zeros_last))
