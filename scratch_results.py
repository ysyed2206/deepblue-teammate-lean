import glob, os, re, chess, chess.pgn
seen, rows = set(), []
for path in sorted(glob.glob(r"C:/Users/uniqu/Downloads/aichessathon-round-*.pgn")):
    m = re.search(r"round-(\d+)", os.path.basename(path))
    g = chess.pgn.read_game(open(path))
    if not m or g is None: continue
    rnd = int(m.group(1))
    if rnd in seen: continue
    seen.add(rnd)
    h = g.headers
    us = chess.WHITE if h.get("White","").lower().startswith("yumo") else chess.BLACK
    res = h.get("Result","*"); ours = "1-0" if us == chess.WHITE else "0-1"
    pts = 1.0 if res == ours else (0.5 if res == "1/2-1/2" else 0.0)
    rows.append((rnd, "W" if us == chess.WHITE else "B", pts))
rows.sort()
def block(lo, hi):
    v = [r for r in rows if lo <= r[0] <= hi]
    if not v: return None
    return len(v), sum(r[2] for r in v), 100*sum(r[2] for r in v)/len(v)
print("%-12s %6s %7s %8s" % ("rounds", "games", "points", "score"))
for lo, hi in ((1,15),(16,30),(31,45),(46,58)):
    b = block(lo, hi)
    if b: print("%-12s %6d %7.1f %7.1f%%" % ("%d-%d" % (lo,hi), *b))
b = block(1,99); print("%-12s %6d %7.1f %7.1f%%" % ("all", *b))
w = [r for r in rows if r[1]=="W"]; bl = [r for r in rows if r[1]=="B"]
print("\nas White: %d games %.1f%%   as Black: %d games %.1f%%"
      % (len(w), 100*sum(r[2] for r in w)/len(w),
         len(bl), 100*sum(r[2] for r in bl)/len(bl)))
