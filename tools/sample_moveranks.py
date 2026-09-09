"""Sample positions where Stockfish scored SEVERAL candidate moves.

WHY (2026-09-09, from the user's own reading of rounds 76 and 77). The
complaint is not that we hang pieces -- it is that we play the second or
third best move, repeatedly, and lose ground to an opponent who plays the
best one. Texel MSE does not measure that: it fits the MAGNITUDE of an
evaluation, and a change can lower it while ranking moves exactly as before.

The Lichess database stores several pvs per position, each with its own cp.
That is a ranked move list from a deep search -- the ground truth for
"did we pick the best move". This writes one record per position:

    <fen> TAB uci:cp uci:cp uci:cp ...

cp is White-relative and is the score AFTER that move.
"""
from __future__ import annotations

import argparse
import io
import json

import zstandard as zstd

SRC = (r"C:/Users/uniqu/AppData/Local/Temp/deepblue-teammate-data"
       r"/lichess_eval/lichess_db_eval.jsonl.zst")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratch_moveranks.txt")
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--stride", type=int, default=11)
    ap.add_argument("--min-depth", type=int, default=22)
    ap.add_argument("--min-moves", type=int, default=3)
    ap.add_argument("--clip", type=int, default=1200)
    args = ap.parse_args()

    kept = seen = 0
    with open(SRC, "rb") as fh, open(args.out, "w", encoding="utf-8") as out:
        stream = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh),
                                  encoding="utf-8")
        for line in stream:
            seen += 1
            if seen % args.stride:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            best = None
            for ev in rec.get("evals") or []:
                if ev.get("depth", 0) >= args.min_depth and (
                        best is None or ev["depth"] > best["depth"]):
                    best = ev
            if best is None:
                continue
            moves = []
            for pv in best.get("pvs") or []:
                if "cp" not in pv or not pv.get("line"):
                    continue
                cp = int(pv["cp"])
                if abs(cp) > args.clip:
                    moves = []
                    break
                moves.append((pv["line"].split()[0], cp))
            if len(moves) < args.min_moves:
                continue
            spread = max(c for _, c in moves) - min(c for _, c in moves)
            if spread < 30:
                continue          # nothing to get wrong; every move is equal
            fen = rec["fen"]
            if len(fen.split()) == 4:
                fen += " 0 1"
            out.write(fen + "\t" + " ".join("%s:%d" % m for m in moves) + "\n")
            kept += 1
            if kept >= args.n:
                break
            if kept % 20_000 == 0:
                print("kept %d / read %d" % (kept, seen), flush=True)
    print("done: %d positions from %d records" % (kept, seen))


if __name__ == "__main__":
    main()
