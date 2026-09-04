"""Convert the downloaded Lichess/Stockfish JSONL positions into the compact
NPZ format ``nnue_lab.train``'s trainer expects (``indices[N,2,32]`` uint16,
``sides[N]`` int, ``targets[N]`` float32 centipawns, side-to-move relative).

Deliberately a lean, direct version of ``nnue_lab.preprocess`` (which builds
a full pilot/final/manifest pipeline for a large, reproducible production
run) -- this is a modest, time-boxed fine-tune dataset, not a production
corpus, so it skips that machinery and goes straight from JSONL to NPZ using
the same encoder (``nnue_lab.features.encode_board`` / ``stm_target``) so
the feature convention is identical either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

from nnue_lab.features import encode_board, normalize_fen, stm_target  # noqa: E402


def load_positions(jsonl_path: Path, clip_cp: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices_list = []
    sides_list = []
    targets_list = []
    skipped_mate = 0
    skipped_error = 0
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("mate") is not None:
                skipped_mate += 1
                continue
            try:
                fen = normalize_fen(row["fen"])
                board = chess.Board(fen)
                encoded = encode_board(board)  # uint16[2,32], white-king-pov/black-king-pov, fixed order
                target = stm_target(int(row["cp"]), board, clip_cp=clip_cp)
            except (ValueError, KeyError):
                skipped_error += 1
                continue
            indices_list.append(encoded)
            sides_list.append(0 if board.turn == chess.WHITE else 1)
            targets_list.append(target)
    indices = np.stack(indices_list).astype(np.uint16)
    sides = np.array(sides_list, dtype=np.int64)
    targets = np.array(targets_list, dtype=np.float32)
    print(f"loaded {len(indices)} positions ({skipped_mate} mate-skipped, {skipped_error} parse errors)")
    return indices, sides, targets


def main() -> None:
    jsonl_path = Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\finetune_data\shard0\selected.jsonl")
    indices, sides, targets = load_positions(jsonl_path)

    rng = np.random.default_rng(20260904)
    n = len(indices)
    perm = rng.permutation(n)
    val_size = max(1000, n // 20)  # ~5% validation
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    out_dir = Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\finetune_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "train.npz", indices=indices[train_idx], sides=sides[train_idx], targets=targets[train_idx])
    np.savez(out_dir / "validation.npz", indices=indices[val_idx], sides=sides[val_idx], targets=targets[val_idx])
    print(f"train: {len(train_idx)}, validation: {len(val_idx)}")
    print(f"wrote {out_dir / 'train.npz'} and {out_dir / 'validation.npz'}")


if __name__ == "__main__":
    main()
