"""How much of its clock does the engine actually spend? No search required."""
import glob, os, chess, chess.pgn
rows = []
for path in sorted(glob.glob(r"C:/Users/uniqu/Downloads/aichessathon-round-*.pgn")):
    g = chess.pgn.read_game(open(path))
    if g is None: continue
    hdr = g.headers
    us = chess.WHITE if hdr.get("White","").lower().startswith("yumo") else chess.BLACK
    node, prev, spends = g, None, []
    while node.variations:
        node = node.variation(0)
        if node.parent.board().turn == us:
            clk = node.clock()
            if clk is not None:
                if prev is not None:
                    spends.append(prev + 0.5 - clk)
                prev = clk
    if not spends: continue
    res = hdr.get("Result","*")
    ours = "1-0" if (us == chess.WHITE) else "0-1"
    outcome = "win" if res == ours else ("draw" if res == "1/2-1/2" else "loss")
    rows.append((os.path.basename(path).split("-")[2][:14], "W" if us==chess.WHITE else "B",
                 outcome, len(spends), sum(spends)/len(spends), min(spends), prev))
print("%-15s %2s %-5s %4s %7s %7s %9s" % ("round","sd","result","mvs","avg s","min s","left s"))
for r in rows:
    print("%-15s %2s %-5s %4d %7.2f %7.2f %9.1f" % r)
if rows:
    import statistics
    print("\nmean time left at game end: %.1f s of 120 (%.0f%% of the clock never spent)"
          % (statistics.mean(r[6] for r in rows),
             100*statistics.mean(r[6] for r in rows)/120))
    print("mean seconds per move:      %.2f s" % statistics.mean(r[4] for r in rows))

# --- breakdown by outcome, and how spending tracks the clock -------------
import statistics
by = {}
for r in rows:
    by.setdefault(r[2], []).append(r)
print("\n%-6s %5s %9s %9s %9s" % ("result", "games", "avg s/mv", "left s", "moves"))
for k in ("win", "draw", "loss"):
    v = by.get(k, [])
    if not v: continue
    print("%-6s %5d %9.2f %9.1f %9.1f" % (k, len(v),
          statistics.mean(x[4] for x in v), statistics.mean(x[6] for x in v),
          statistics.mean(x[3] for x in v)))

# Does the engine speed up or slow down as its clock shrinks? Compare the
# average spend in the first third of our moves against the last third.
import glob, os
first_thirds, last_thirds = [], []
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
    if len(sp) >= 12:
        t = len(sp)//3
        first_thirds.append(statistics.mean(sp[:t]))
        last_thirds.append(statistics.mean(sp[-t:]))
print("\nspend in first third of our moves: %.2f s" % statistics.mean(first_thirds))
print("spend in last  third of our moves: %.2f s" % statistics.mean(last_thirds))
