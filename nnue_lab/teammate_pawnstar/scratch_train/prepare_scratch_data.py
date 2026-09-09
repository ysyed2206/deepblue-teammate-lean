"""Convert the downloaded Lichess/Stockfish JSONL positions into the NPZ
format nnue_lab.train's trainer expects. Same encoder/logic as
finetune/prepare_finetune_data.py, pointed at the larger from-scratch
dataset and its own output location -- kept as a separate copy rather than
a shared import so the two lanes (non-shippable fine-tune experiment vs.
this compliant from-scratch lane) never accidentally share state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

from nnue_lab.features import encode_board, normalize_fen, stm_target  # noqa: E402


def load_positions(jsonl_paths: list[Path], clip_cp: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices_list = []
    sides_list = []
    targets_list = []
    skipped_mate = 0
    skipped_error = 0
    skipped_check = 0
    for jsonl_path in jsonl_paths:
        count_before = len(indices_list)
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("mate") is not None:
                    skipped_mate += 1
                    continue
                try:
                    fen = normalize_fen(row["fen"])
                    board = chess.Board(fen)
                    # NNUE evaluation represents a static, "quiet" judgement
                    # of a position -- a side to move in check is inherently
                    # mid-tactic (some reply is forced), not the kind of
                    # position the evaluation function is meant to score.
                    # Cheap to filter; full SEE-based tactical-volatility
                    # filtering (used by real large-scale NNUE training) is
                    # not attempted here.
                    if board.is_check():
                        skipped_check += 1
                        continue
                    encoded = encode_board(board)
                    target = stm_target(int(row["cp"]), board, clip_cp=clip_cp)
                except (ValueError, KeyError):
                    skipped_error += 1
                    continue
                indices_list.append(encoded)
                sides_list.append(0 if board.turn == chess.WHITE else 1)
                targets_list.append(target)
        print(f"  {jsonl_path.parent.name}: {len(indices_list) - count_before} positions")
    indices = np.stack(indices_list).astype(np.uint16)
    sides = np.array(sides_list, dtype=np.int64)
    targets = np.array(targets_list, dtype=np.float32)
    print(f"loaded {len(indices)} positions total "
          f"({skipped_mate} mate-skipped, {skipped_check} in-check-skipped, {skipped_error} parse errors)")
    return indices, sides, targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\scratch_data_big"),
        help="directory containing shard*/selected.jsonl (as written by download_data.py)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\scratch_data_big_npz"),
        help="where to write train.npz and validation.npz",
    )
    args = parser.parse_args()

    jsonl_paths = sorted(args.input_dir.glob("shard*/selected.jsonl"))
    print(f"found {len(jsonl_paths)} shards in {args.input_dir}")
    if not jsonl_paths:
        raise SystemExit(f"no shard*/selected.jsonl found under {args.input_dir}")
    indices, sides, targets = load_positions(jsonl_paths)

    rng = np.random.default_rng(20260905)
    n = len(indices)
    perm = rng.permutation(n)
    val_size = max(2000, n // 20)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "train.npz", indices=indices[train_idx], sides=sides[train_idx], targets=targets[train_idx])
    np.savez(out_dir / "validation.npz", indices=indices[val_idx], sides=sides[val_idx], targets=targets[val_idx])
    print(f"train: {len(train_idx)}, validation: {len(val_idx)}")
    print(f"wrote {out_dir / 'train.npz'} and {out_dir / 'validation.npz'}")


if __name__ == "__main__":
    main()
