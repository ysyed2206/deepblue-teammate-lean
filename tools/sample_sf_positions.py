"""Sample (FEN, Stockfish cp) pairs from the Lichess evaluation database.

WHY THIS EXISTS (2026-09-09). Every eval investigation in this project has
used our own engine as the referee, which is blind to exactly what our
evaluation is blind to. Rounds 72, 76 and 78 each reported "zero errors >=100cp"
in games we were checkmated in. The Lichess database carries deep Stockfish
scores for millions of positions, so it can referee our static eval directly.

This is annotation data, not engine code or weights: nothing from Stockfish is
shipped, and nothing here runs at match time.

Lichess stores cp White-relative. We keep that convention throughout.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import zstandard as zstd

SRC = Path(r"C:/Users/uniqu/AppData/Local/Temp/deepblue-teammate-data"
           r"/lichess_eval/lichess_db_eval.jsonl.zst")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratch_sf_positions.txt")
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--stride", type=int, default=7, help="decorrelate consecutive records")
    ap.add_argument("--min-depth", type=int, default=20)
    ap.add_argument("--clip", type=int, default=1500, help="skip |cp| above this")
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
            evals = rec.get("evals") or []
            best = None
            for ev in evals:
                if ev.get("depth", 0) >= args.min_depth and (
                        best is None or ev["depth"] > best["depth"]):
                    best = ev
            if best is None:
                continue
            pvs = best.get("pvs") or []
            if not pvs or "cp" not in pvs[0]:
                continue                      # mate scores carry no cp
            cp = int(pvs[0]["cp"])
            if abs(cp) > args.clip:
                continue
            fen = rec["fen"]
            if len(fen.split()) == 4:         # lichess omits halfmove/fullmove
                fen += " 0 1"
            out.write("%d\t%s\n" % (cp, fen))
            kept += 1
            if kept >= args.n:
                break
            if kept % 50_000 == 0:
                print("kept %d / read %d" % (kept, seen), flush=True)
    print("done: kept %d positions from %d records -> %s" % (kept, seen, args.out))


if __name__ == "__main__":
    main()
