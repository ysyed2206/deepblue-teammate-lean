"""Export and validate the participant-owned exact-int16 NNUE v0 format."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nnue_lab.features import NUM_FEATURES, PADDING_FEATURE
from nnue_lab.model import EVAL_SCALE_CP, load_checkpoint

FORMAT = "deepblue-nnue-int16-v0"
QA_DEFAULT = 255
QB_DEFAULT = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def quantise_checked(array: np.ndarray, scale: int, name: str) -> np.ndarray:
    rounded = np.rint(array.astype(np.float64) * scale)
    if rounded.min() < np.iinfo(np.int16).min or rounded.max() > np.iinfo(np.int16).max:
        raise OverflowError(f"{name} does not fit int16 at scale {scale}")
    return rounded.astype(np.int16)


def load_quantised(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=False)
    format_name = str(loaded["format"].item())
    if format_name != FORMAT:
        raise ValueError(f"unsupported quantised format: {format_name}")
    result = {name: loaded[name] for name in loaded.files}
    width = int(result["width"].item())
    if result["feature_weights"].shape != (NUM_FEATURES, width):
        raise ValueError("feature weight shape/header mismatch")
    if result["feature_bias"].shape != (width,):
        raise ValueError("feature bias shape/header mismatch")
    if result["output_weights"].shape != (2 * width,):
        raise ValueError("output weight shape/header mismatch")
    return result


def symmetric_round_divide(numerator: np.ndarray, denominator: int) -> np.ndarray:
    absolute = np.abs(numerator)
    quotient = (absolute + denominator // 2) // denominator
    return np.where(numerator < 0, -quotient, quotient)


def quantised_predict(
    model: dict[str, Any], indices: np.ndarray, sides: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    width = int(model["width"].item())
    qa = int(model["qa"].item())
    qb = int(model["qb"].item())
    evaluation_scale = int(model["evaluation_scale_cp"].item())
    feature_weights = model["feature_weights"]
    padded_weights = np.zeros((NUM_FEATURES + 1, width), dtype=np.int16)
    padded_weights[:NUM_FEATURES] = feature_weights
    feature_bias = model["feature_bias"].astype(np.int32)
    output_weights = model["output_weights"].astype(np.int64)
    output_bias = int(model["output_bias"].item())
    denominator = qb * qa * qa
    result = np.empty(len(indices), dtype=np.int32)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end].astype(np.int64, copy=False)
        if np.any(batch_indices > PADDING_FEATURE):
            raise ValueError("feature index exceeds padding sentinel")
        accumulators = padded_weights[batch_indices].sum(axis=2, dtype=np.int32)
        accumulators += feature_bias[None, None, :]
        activated = np.clip(accumulators, 0, qa).astype(np.int64)
        activated *= activated
        batch_sides = sides[start:end].astype(np.int64, copy=False)
        batch_index = np.arange(end - start)
        stm = activated[batch_index, batch_sides]
        non_stm = activated[batch_index, 1 - batch_sides]
        numerator = (
            output_bias * qa * qa
            + stm @ output_weights[:width]
            + non_stm @ output_weights[width:]
        )
        result[start:end] = symmetric_round_divide(
            numerator * evaluation_scale, denominator
        ).astype(np.int32)
    return result


@torch.no_grad()
def float_predict(
    checkpoint: Path, indices: np.ndarray, sides: np.ndarray, batch_size: int
) -> tuple[np.ndarray, dict[str, object]]:
    model, metadata = load_checkpoint(checkpoint)
    predictions = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        tensor_indices = torch.from_numpy(indices[start:end].astype(np.int64, copy=True))
        tensor_sides = torch.from_numpy(sides[start:end].astype(np.int64, copy=True))
        predictions[start:end] = model(tensor_indices, tensor_sides).numpy()
    return predictions, metadata


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median_cp": float(np.percentile(values, 50)),
        "p95_cp": float(np.percentile(values, 95)),
        "p99_cp": float(np.percentile(values, 99)),
        "maximum_cp": float(np.max(values)),
        "mean_cp": float(np.mean(values)),
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    model, checkpoint_metadata = load_checkpoint(args.checkpoint)
    state = model.state_dict()
    feature_weights = quantise_checked(
        state["feature_weights.weight"].cpu().numpy(), args.qa, "feature_weights"
    )
    feature_bias = quantise_checked(
        state["feature_bias"].cpu().numpy(), args.qa, "feature_bias"
    )
    output_weights = quantise_checked(
        state["output.weight"].cpu().numpy().reshape(-1), args.qb, "output_weights"
    )
    output_bias = quantise_checked(
        state["output.bias"].cpu().numpy(), args.qb, "output_bias"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        format=np.asarray(FORMAT),
        version=np.asarray(0, dtype=np.int32),
        width=np.asarray(model.width, dtype=np.int32),
        num_features=np.asarray(NUM_FEATURES, dtype=np.int32),
        perspectives=np.asarray(2, dtype=np.int32),
        max_pieces=np.asarray(32, dtype=np.int32),
        qa=np.asarray(args.qa, dtype=np.int32),
        qb=np.asarray(args.qb, dtype=np.int32),
        evaluation_scale_cp=np.asarray(int(EVAL_SCALE_CP), dtype=np.int32),
        feature_weights=feature_weights,
        feature_bias=feature_bias,
        output_weights=output_weights,
        output_bias=output_bias,
    )
    quantised_model = load_quantised(args.output)
    heldout = np.load(args.heldout, allow_pickle=False)
    indices = heldout["indices"]
    sides = heldout["sides"]
    targets = heldout["targets"].astype(np.float32)
    if len(indices) < 10_000:
        raise RuntimeError("quantisation gate requires at least 10,000 held-out FENs")
    started = time.perf_counter()
    float_cp, _ = float_predict(args.checkpoint, indices, sides, args.batch_size)
    integer_cp = quantised_predict(quantised_model, indices, sides, args.batch_size)
    elapsed = time.perf_counter() - started
    quantisation_error = np.abs(integer_cp.astype(np.float64) - float_cp.astype(np.float64))
    float_teacher_error = np.abs(float_cp - targets)
    integer_teacher_error = np.abs(integer_cp.astype(np.float32) - targets)
    metrics: dict[str, Any] = {
        "format": "deepblue-nnue-quantisation-report-v0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "derivation": (
            "A=clip(round(QA*b)+sum(round(QA*W)),0,QA); "
            "cp=round_symmetric(S*(round(QB*c)*QA^2+sum(round(QB*v)*A^2))/(QB*QA^2))"
        ),
        "rounding": "weights use round-to-nearest-even; final integer cp uses symmetric nearest",
        "qa": args.qa,
        "qb": args.qb,
        "evaluation_scale_cp": int(EVAL_SCALE_CP),
        "width": model.width,
        "deploy_parameter_count": model.deploy_parameter_count(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "quantised_model": str(args.output.resolve()),
        "quantised_model_sha256": sha256_file(args.output),
        "quantised_model_bytes": args.output.stat().st_size,
        "heldout": str(args.heldout.resolve()),
        "heldout_sha256": sha256_file(args.heldout),
        "heldout_count": len(indices),
        "float_teacher_mae_cp": float(np.mean(float_teacher_error)),
        "quantised_teacher_mae_cp": float(np.mean(integer_teacher_error)),
        "quantisation_absolute_error": percentile_summary(quantisation_error),
        "crosscheck_seconds": elapsed,
        "crosscheck_examples_per_second": len(indices) / elapsed,
        "integer_ranges": {
            "feature_weight_min": int(feature_weights.min()),
            "feature_weight_max": int(feature_weights.max()),
            "feature_bias_min": int(feature_bias.min()),
            "feature_bias_max": int(feature_bias.max()),
            "output_weight_min": int(output_weights.min()),
            "output_weight_max": int(output_weights.max()),
            "output_bias": int(output_bias.item()),
        },
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--qa", type=int, default=QA_DEFAULT)
    parser.add_argument("--qb", type=int, default=QB_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    lab = Path(__file__).resolve().parent
    if args.qa <= 0 or args.qb <= 0 or args.batch_size <= 0 or args.threads <= 0:
        parser.error("quantisation scales, batch size, and threads must be positive")
    if args.output.suffix.lower() != ".npz":
        parser.error("quantised output path must end in .npz")
    for output in (args.output, args.metrics):
        if not output.resolve().is_relative_to(lab):
            parser.error("quantised model and metrics must stay inside nnue_lab")
    return args


if __name__ == "__main__":
    export(parse_args())
