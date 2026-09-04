"""Evaluate production float/Q1, accepted HCE, and optional H128 on one set."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PRODUCTION_DIR = Path(__file__).resolve().parent
LAB_DIR = PRODUCTION_DIR.parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(PRODUCTION_DIR / ".numba_cache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from nnue_lab.production.artifacts import (  # noqa: E402
    BASELINE_FLOAT_FORMAT,
    LoadedFloatModel,
    load_float_model,
)
from nnue_lab.production.dataset import (  # noqa: E402
    load_encoded_npz,
    reject_pristine_before_selection,
    sha256_file,
)
from nnue_lab.production.export import (  # noqa: E402
    float_predict,
    load_quantized,
    pearson_correlation,
    probability_loss_numpy,
    quantized_predict,
)
from nnue_lab.production.features import encode_board as encode_chess768  # noqa: E402


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_fen(fen: str) -> str:
    fields = fen.strip().split()
    if len(fields) == 4:
        fields.extend(("0", "1"))
    if len(fields) != 6:
        raise ValueError(f"FEN has {len(fields)} fields rather than four or six")
    return " ".join(fields)


def load_fen_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, str):
                record: dict[str, Any] = {"fen": value}
            elif isinstance(value, dict) and "fen" in value:
                record = dict(value)
            else:
                raise ValueError(f"{path}:{line_number}: expected a FEN record")
            record["fen"] = normalize_fen(str(record["fen"]))
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no FEN records")
    return records


def error_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float | None]:
    signed_errors = predictions.astype(np.float64) - targets.astype(np.float64)
    errors = np.abs(signed_errors)
    return {
        "probability_mse": probability_loss_numpy(predictions, targets),
        "mae_cp": float(errors.mean()),
        "signed_bias_cp": float(signed_errors.mean()),
        "pearson_correlation": pearson_correlation(predictions, targets),
        "median_absolute_error_cp": float(np.percentile(errors, 50)),
        "p95_absolute_error_cp": float(np.percentile(errors, 95)),
        "p99_absolute_error_cp": float(np.percentile(errors, 99)),
        "maximum_absolute_error_cp": float(errors.max()),
    }


def subset_mae(
    predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> float | None:
    if not np.any(mask):
        return None
    return float(
        np.mean(
            np.abs(
                predictions[mask].astype(np.float64) - targets[mask].astype(np.float64)
            )
        )
    )


def validate_fen_alignment(
    records: list[dict[str, Any]],
    indices: np.ndarray,
    sides: np.ndarray,
    targets: np.ndarray,
) -> None:
    if len(records) != len(indices):
        raise ValueError("FEN JSONL and encoded NPZ have different row counts")
    for row, record in enumerate(records):
        board = chess.Board(str(record["fen"]))
        side = 0 if board.turn == chess.WHITE else 1
        if side != int(sides[row]):
            raise ValueError(f"row {row}: FEN side-to-move and sides array disagree")
        encoded = encode_chess768(board)
        if not np.array_equal(encoded, indices[row]):
            raise ValueError(f"row {row}: FEN and Chess768 feature rows disagree")
        if "target_cp" in record and int(record["target_cp"]) != int(targets[row]):
            raise ValueError(f"row {row}: FEN record and NPZ teacher target disagree")


def hce_predict(records: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    # fastsearch4 is the accepted search wrapper; these evaluator symbols are
    # imported there from the shared accepted HCE implementation.
    from deepblue.fastcore import from_fen
    from deepblue.fastsearch4 import EG_TABLE, MG_TABLE, PHASE_TABLE, evaluate

    result = np.empty(len(records), dtype=np.int32)
    warm_bb, _, _, warm_state = from_fen(str(records[0]["fen"]))
    evaluate(warm_bb, warm_state, MG_TABLE, EG_TABLE, PHASE_TABLE)
    started = time.perf_counter()
    for row, record in enumerate(records):
        bitboards, _, _, state = from_fen(str(record["fen"]))
        result[row] = evaluate(bitboards, state, MG_TABLE, EG_TABLE, PHASE_TABLE)
    return result, time.perf_counter() - started


@torch.no_grad()
def baseline_predict(
    loaded: LoadedFloatModel,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
) -> np.ndarray:
    if loaded.format_name != BASELINE_FLOAT_FORMAT or loaded.num_features != 6144:
        raise ValueError("baseline checkpoint must be the preserved 6144-row v0 format")
    from nnue_lab.features import encode_board as encode_chessbuckets

    result = np.empty(len(records), dtype=np.float32)
    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        boards = [chess.Board(str(record["fen"])) for record in records[start:end]]
        encoded = np.stack([encode_chessbuckets(board) for board in boards])
        sides = np.asarray(
            [0 if board.turn == chess.WHITE else 1 for board in boards], dtype=np.int64
        )
        tensor_indices = torch.from_numpy(encoded.astype(np.int64, copy=False))
        tensor_sides = torch.from_numpy(sides)
        result[start:end] = loaded.model(tensor_indices, tensor_sides).cpu().numpy()
    return result


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    guarded_paths = [args.dataset]
    if args.fens is not None:
        guarded_paths.append(args.fens)
    reject_pristine_before_selection(guarded_paths, args.final_selection_locked)
    arrays = load_encoded_npz(
        args.dataset,
        legacy_chessbuckets_mod768=args.legacy_chessbuckets_mod768,
    )
    production = load_float_model(args.checkpoint)
    quantized = load_quantized(args.quantized)
    if quantized.width != production.width:
        raise ValueError("float and Q1 widths differ")
    if quantized.metadata.get("checkpoint_sha256") != sha256_file(args.checkpoint):
        raise ValueError("Q1 artifact was not exported from the supplied float checkpoint")

    records = None if args.fens is None else load_fen_records(args.fens)
    if records is not None:
        validate_fen_alignment(records, arrays.indices, arrays.sides, arrays.targets)
    if (not args.skip_hce or args.baseline_checkpoint is not None) and records is None:
        raise ValueError("HCE and H128 comparisons require --fens")

    started = time.perf_counter()
    float_cp = float_predict(
        production, arrays.indices, arrays.sides, batch_size=args.batch_size
    )
    float_seconds = time.perf_counter() - started
    started = time.perf_counter()
    quantized_cp = quantized_predict(
        quantized, arrays.indices, arrays.sides, batch_size=args.batch_size
    )
    quantized_seconds = time.perf_counter() - started

    predictions: dict[str, np.ndarray] = {
        "target_cp": arrays.targets.astype(np.float32),
        "production_float_cp": float_cp,
        "production_q1_cp": quantized_cp,
        "sides": arrays.sides,
    }
    metrics: dict[str, Any] = {
        "production_float": error_metrics(float_cp, arrays.targets),
        "production_q1": error_metrics(quantized_cp, arrays.targets),
    }
    timing: dict[str, Any] = {
        "production_float_full_pass_seconds": float_seconds,
        "production_float_examples_per_second": arrays.count / float_seconds,
        "production_q1_numpy_reference_full_pass_seconds": quantized_seconds,
        "production_q1_numpy_reference_examples_per_second": arrays.count
        / quantized_seconds,
    }

    if not args.skip_hce:
        assert records is not None
        hce_cp, hce_seconds = hce_predict(records)
        predictions["hce_cp"] = hce_cp
        metrics["accepted_deepblue_hce"] = error_metrics(hce_cp, arrays.targets)
        timing["hce_full_pass_seconds_including_fen_parse"] = hce_seconds
        timing["hce_fens_per_second_including_fen_parse"] = arrays.count / hce_seconds

    if args.baseline_checkpoint is not None:
        assert records is not None
        baseline = load_float_model(args.baseline_checkpoint)
        started = time.perf_counter()
        baseline_cp = baseline_predict(baseline, records, batch_size=args.batch_size)
        baseline_seconds = time.perf_counter() - started
        predictions["baseline_h128_float_cp"] = baseline_cp
        metrics["preserved_h128_float"] = error_metrics(baseline_cp, arrays.targets)
        timing["baseline_h128_full_pass_seconds_including_fen_encoding"] = (
            baseline_seconds
        )

    if "accepted_deepblue_hce" in metrics:
        hce_mae = float(metrics["accepted_deepblue_hce"]["mae_cp"])
        metrics["production_float_improvement_over_hce_mae_fraction"] = (
            hce_mae - float(metrics["production_float"]["mae_cp"])
        ) / hce_mae
        metrics["production_q1_improvement_over_hce_mae_fraction"] = (
            hce_mae - float(metrics["production_q1"]["mae_cp"])
        ) / hce_mae

    subsets: dict[str, Any] = {}
    if records is not None and all("quiet" in record for record in records):
        quiet = np.asarray([bool(record["quiet"]) for record in records])
        subsets["quiet_count"] = int(quiet.sum())
        subsets["nonquiet_count"] = int((~quiet).sum())
        for name, values in predictions.items():
            if name in ("target_cp", "sides"):
                continue
            subsets[f"{name}_quiet_mae_cp"] = subset_mae(
                values, arrays.targets, quiet
            )
            subsets[f"{name}_nonquiet_mae_cp"] = subset_mae(
                values, arrays.targets, ~quiet
            )

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.predictions, **predictions)
    report: dict[str, Any] = {
        "format": "deepblue-perspective-chess768-heldout-comparison-v1",
        "created_utc": utc_now(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "fens": None if args.fens is None else str(args.fens.resolve()),
        "fens_sha256": None if args.fens is None else sha256_file(args.fens),
        "count": arrays.count,
        "target": "clipped Stockfish cp from side-to-move perspective",
        "architecture_selection_declared_locked": args.final_selection_locked,
        "is_pristine": any("pristine" in str(path).lower() for path in guarded_paths),
        "metrics": metrics,
        "subsets": subsets,
        "timing": timing,
        "production_checkpoint": str(args.checkpoint.resolve()),
        "production_checkpoint_sha256": sha256_file(args.checkpoint),
        "production_checkpoint_metadata": production.metadata,
        "q1_model": str(args.quantized.resolve()),
        "q1_model_sha256": sha256_file(args.quantized),
        "baseline_checkpoint": (
            None
            if args.baseline_checkpoint is None
            else str(args.baseline_checkpoint.resolve())
        ),
        "baseline_checkpoint_sha256": (
            None
            if args.baseline_checkpoint is None
            else sha256_file(args.baseline_checkpoint)
        ),
        "accepted_hce_symbol": (
            "deepblue.fastsearch4.evaluate alias of "
            "deepblue.fastsearch.evaluate(bb,st,MG_TABLE,EG_TABLE,PHASE_TABLE)"
        ),
        "accepted_hce_implementation_sha256": sha256_file(
            LAB_DIR.parent / "deepblue" / "fastsearch.py"
        ),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fens", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-hce", action="store_true")
    parser.add_argument("--legacy-chessbuckets-mod768", action="store_true")
    parser.add_argument("--final-selection-locked", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.quantized = args.quantized.resolve()
    args.fens = None if args.fens is None else args.fens.resolve()
    args.baseline_checkpoint = (
        None
        if args.baseline_checkpoint is None
        else args.baseline_checkpoint.resolve()
    )
    args.report = args.report.resolve()
    args.predictions = args.predictions.resolve()
    for output in (args.report, args.predictions):
        if not output.is_relative_to(LAB_DIR):
            parser.error("evaluation outputs must stay inside nnue_lab")
    if args.predictions.suffix.lower() != ".npz":
        parser.error("predictions path must end in .npz")
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch size and threads must be positive")
    return args


if __name__ == "__main__":
    run_evaluation(parse_args())
