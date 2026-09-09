"""Convert a slice of Lichess's official Stockfish-evaluation database
(lichess_db_eval.jsonl.zst, https://database.lichess.org/, CC0) into the same
NPZ format nnue_lab.train's trainer already expects -- same encoder/filters as
prepare_scratch_data.py, pointed at this much larger, differently-shaped
source instead of the hand-selected shards.

Each source line is one FEN with one or more `evals` entries (different
engine strengths/depths that analysed it); each eval carries one or more
`pvs` (principal variations). We take the *deepest* eval entry (most reliable
label) and that eval's first ("pvs[0]", i.e. the actual best-line) score.

Streams the .zst directly -- the file is ~18GB compressed and reads out to
roughly 10x that, so nothing here materializes the whole thing in memory.
A `--limit` caps how many usable positions are collected, for the cheap
first-pass test (tens of millions) before committing to the full ~395M.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

from nnue_lab.features import encode_board, normalize_fen, stm_target  # noqa: E402


def iter_lines(zst_path: Path):
    with zst_path.open("rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        buffer = b""
        while True:
            chunk = reader.read(1 << 20)
            if not chunk:
                break
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                yield line
        if buffer:
            yield buffer


def best_eval(row: dict) -> tuple[int | None, int | None]:
    """Return (cp, mate) from the deepest eval's principal line."""
    evals = row.get("evals")
    if not evals:
        return None, None
    deepest = max(evals, key=lambda e: e.get("depth", 0))
    pvs = deepest.get("pvs")
    if not pvs:
        return None, None
    pv = pvs[0]
    return pv.get("cp"), pv.get("mate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zst-path", type=Path,
                     default=Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data"
                                  r"\lichess_eval\lichess_db_eval.jsonl.zst"))
    ap.add_argument("--out-dir", type=Path,
                     default=Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data"
                                  r"\lichess_eval_npz"))
    ap.add_argument("--limit", type=int, default=30_000_000,
                     help="stop after this many USABLE (post-filter) positions")
    ap.add_argument("--clip-cp", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--report-every", type=int, default=1_000_000)
    args = ap.parse_args()

    indices_list = []
    sides_list = []
    targets_list = []
    seen = 0
    skipped_no_eval = 0
    skipped_mate = 0
    skipped_check = 0
    skipped_error = 0

    for line in iter_lines(args.zst_path):
        if not line:
            continue
        seen += 1
        if seen % args.report_every == 0:
            print(f"  scanned {seen:,} lines, kept {len(indices_list):,}", flush=True)
        row = json.loads(line)
        cp, mate = best_eval(row)
        if cp is None:
            skipped_no_eval += 1
            continue
        if mate is not None:
            skipped_mate += 1
            continue
        try:
            fen = normalize_fen(row["fen"])
            board = chess.Board(fen)
            if board.is_check():
                skipped_check += 1
                continue
            encoded = encode_board(board)
            target = stm_target(int(cp), board, clip_cp=args.clip_cp)
        except (ValueError, KeyError):
            skipped_error += 1
            continue
        indices_list.append(encoded)
        sides_list.append(0 if board.turn == chess.WHITE else 1)
        targets_list.append(target)
        if len(indices_list) >= args.limit:
            break

    indices = np.stack(indices_list).astype(np.uint16)
    sides = np.array(sides_list, dtype=np.int64)
    targets = np.array(targets_list, dtype=np.float32)
    print(f"scanned {seen:,} lines total; kept {len(indices):,} "
          f"({skipped_no_eval:,} no-eval, {skipped_mate:,} mate-skipped, "
          f"{skipped_check:,} in-check-skipped, {skipped_error:,} parse errors)")

    rng = np.random.default_rng(args.seed)
    n = len(indices)
    perm = rng.permutation(n)
    val_size = max(2000, n // 20)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "train.npz", indices=indices[train_idx], sides=sides[train_idx],
              targets=targets[train_idx])
    np.savez(args.out_dir / "validation.npz", indices=indices[val_idx], sides=sides[val_idx],
              targets=targets[val_idx])
    print(f"train: {len(train_idx):,}, validation: {len(val_idx):,}")
    print(f"wrote {args.out_dir / 'train.npz'} and {args.out_dir / 'validation.npz'}")


if __name__ == "__main__":
    main()
