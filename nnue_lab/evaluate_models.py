"""Compare accepted Deep Blue HCE, float NNUE, and int NNUE on one held-out set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

LAB_DIR = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(LAB_DIR / ".numba_cache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nnue_lab.export import (  # noqa: E402
    float_predict,
    load_quantised,
    quantised_predict,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mae(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    return float(np.mean(np.abs(predictions.astype(np.float64) - targets.astype(np.float64))))


def error_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = np.abs(predictions.astype(np.float64) - targets.astype(np.float64))
    return {
        "mae_cp": float(errors.mean()),
        "median_absolute_error_cp": float(np.percentile(errors, 50)),
        "p95_absolute_error_cp": float(np.percentile(errors, 95)),
        "p99_absolute_error_cp": float(np.percentile(errors, 99)),
    }


def load_fen_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def hce_predict(records: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    from deepblue.fastcore import from_fen
    from deepblue.fastsearch import EG_TABLE, MG_TABLE, PHASE_TABLE, evaluate

    predictions = np.empty(len(records), dtype=np.int32)
    # Warm the exact accepted evaluator before timing the complete held-out pass.
    warm_bb, _, _, warm_st = from_fen(str(records[0]["fen"]))
    evaluate(warm_bb, warm_st, MG_TABLE, EG_TABLE, PHASE_TABLE)
    started = time.perf_counter()
    for index, record in enumerate(records):
        bb, _, _, st = from_fen(str(record["fen"]))
        predictions[index] = evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)
    return predictions, time.perf_counter() - started


def evaluate_all(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    heldout = np.load(args.heldout, allow_pickle=False)
    indices = heldout["indices"]
    sides = heldout["sides"]
    targets = heldout["targets"].astype(np.int32)
    records = load_fen_records(args.fens)
    if len(records) != len(indices):
        raise ValueError("held-out NPZ and FEN JSONL have different row counts")
    record_targets = np.asarray([int(record["target_cp"]) for record in records], dtype=np.int32)
    if not np.array_equal(targets, record_targets):
        raise ValueError("held-out NPZ and FEN JSONL target order differs")

    float_cp, checkpoint_metadata = float_predict(
        args.checkpoint, indices, sides, args.batch_size
    )
    quantised_model = load_quantised(args.quantised)
    integer_cp = quantised_predict(quantised_model, indices, sides, args.batch_size)
    hce_cp, hce_seconds = hce_predict(records)
    quiet = np.asarray([bool(record["quiet"]) for record in records])
    white_to_move = sides == 0

    hce_metrics = error_metrics(hce_cp, targets)
    float_metrics = error_metrics(float_cp, targets)
    integer_metrics = error_metrics(integer_cp, targets)
    report: dict[str, Any] = {
        "format": "deepblue-nnue-heldout-comparison-v0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "heldout_count": len(indices),
        "heldout_npz": str(args.heldout.resolve()),
        "heldout_fens": str(args.fens.resolve()),
        "target": "clipped Stockfish cp from side-to-move perspective",
        "hce": hce_metrics,
        "float_nnue": float_metrics,
        "quantised_nnue": integer_metrics,
        "float_improvement_over_hce_mae_fraction": (
            hce_metrics["mae_cp"] - float_metrics["mae_cp"]
        )
        / hce_metrics["mae_cp"],
        "quantised_improvement_over_hce_mae_fraction": (
            hce_metrics["mae_cp"] - integer_metrics["mae_cp"]
        )
        / hce_metrics["mae_cp"],
        "subsets": {
            "quiet_count": int(quiet.sum()),
            "general_nonquiet_count": int((~quiet).sum()),
            "white_to_move_count": int(white_to_move.sum()),
            "black_to_move_count": int((~white_to_move).sum()),
            "hce_quiet_mae_cp": mae(hce_cp, targets, quiet),
            "float_nnue_quiet_mae_cp": mae(float_cp, targets, quiet),
            "quantised_nnue_quiet_mae_cp": mae(integer_cp, targets, quiet),
            "hce_nonquiet_mae_cp": mae(hce_cp, targets, ~quiet),
            "float_nnue_nonquiet_mae_cp": mae(float_cp, targets, ~quiet),
            "quantised_nnue_nonquiet_mae_cp": mae(integer_cp, targets, ~quiet),
        },
        "hce_full_pass_seconds": hce_seconds,
        "hce_fens_per_second_including_fen_parse": len(indices) / hce_seconds,
        "accepted_hce_symbol": (
            "deepblue.fastsearch.evaluate(bb,st,MG_TABLE,EG_TABLE,PHASE_TABLE)"
        ),
        "accepted_hce_source_sha256": sha256_file(
            Path(__file__).resolve().parent.parent / "deepblue" / "fastsearch.py"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "quantised_model": str(args.quantised.resolve()),
        "quantised_model_sha256": sha256_file(args.quantised),
    }
    np.savez(
        args.predictions,
        target_cp=targets.astype(np.int16),
        hce_cp=hce_cp,
        float_nnue_cp=float_cp,
        quantised_nnue_cp=integer_cp,
        quiet=quiet,
        sides=sides,
    )
    report["predictions"] = str(args.predictions.resolve())
    report["predictions_sha256"] = sha256_file(args.predictions)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quantised", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--fens", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    for output in (args.report, args.predictions):
        if not output.resolve().is_relative_to(LAB_DIR):
            parser.error("reports and predictions must stay inside nnue_lab")
    if args.predictions.suffix.lower() != ".npz":
        parser.error("predictions output must end in .npz")
    return args


if __name__ == "__main__":
    evaluate_all(parse_args())

