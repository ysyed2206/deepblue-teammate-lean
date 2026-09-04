"""Export and cross-check the exact donor-compatible int16 Q1 format."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nnue_lab.production.artifacts import (
    PRODUCTION_FLOAT_FORMAT,
    LoadedFloatModel,
    load_float_model,
)
from nnue_lab.production.dataset import (
    EncodedArrays,
    fingerprint_dataset,
    load_encoded_npz,
    reject_pristine_before_selection,
    sha256_file,
)
from nnue_lab.production.features import NUM_FEATURES, PADDING_FEATURE
from nnue_lab.production.model import (
    EVALUATION_SCALE_CP,
    Q1_INTEGER_FORMULA,
    QA_DEFAULT,
    QB_DEFAULT,
    QUANTIZED_FORMAT,
    QUANTIZED_VERSION,
    QuantizedParameters,
    evaluate_quantized_numpy,
    load_quantized_npz,
    quantise_int16,
    save_quantized_npz,
)

LAB_DIR = Path(__file__).resolve().parents[1]
Q1_FORMAT = QUANTIZED_FORMAT
Q1_VERSION = QUANTIZED_VERSION
QuantizedArtifact = QuantizedParameters


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def quantise_checked(array: np.ndarray, scale: int, name: str) -> np.ndarray:
    return np.ascontiguousarray(quantise_int16(array, scale, name))


def load_quantized(path: Path) -> QuantizedArtifact:
    """Use the identical loader and metadata contract consumed by the runtime."""
    return load_quantized_npz(path)


def truncating_divide_array(numerator: np.ndarray, denominator: int) -> np.ndarray:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    absolute = np.abs(numerator)
    quotient = absolute // denominator
    return np.where(numerator < 0, -quotient, quotient)


def quantized_predict_from_accumulators(
    artifact: QuantizedArtifact, accumulators: np.ndarray, sides: np.ndarray
) -> np.ndarray:
    """Vectorized Q1 tail, with the runtime's two signed truncating divisions."""
    if accumulators.ndim != 3 or accumulators.shape[1:] != (2, artifact.width):
        raise ValueError("accumulators must have shape [N,2,width]")
    if accumulators.dtype != np.int32:
        raise TypeError("accumulators must be int32")
    if sides.shape != (len(accumulators),) or not np.all((sides == 0) | (sides == 1)):
        raise ValueError("sides must contain one zero/one per accumulator pair")
    activated = np.clip(accumulators, 0, artifact.qa).astype(np.int64)
    activated *= activated
    rows = np.arange(len(sides))
    batch_sides = sides.astype(np.int64, copy=False)
    weights = artifact.output_weights.astype(np.int64)
    dot = (
        activated[rows, batch_sides] @ weights[: artifact.width]
        + activated[rows, 1 - batch_sides] @ weights[artifact.width :]
    )
    dequantized = truncating_divide_array(dot, artifact.qa) + int(artifact.output_bias)
    return truncating_divide_array(
        dequantized * artifact.evaluation_scale_cp, artifact.qa * artifact.qb
    )


def quantized_predict(
    artifact: QuantizedArtifact,
    indices: np.ndarray,
    sides: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Vectorized reference for the exact Q1 integer arithmetic."""
    artifact.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if indices.dtype != np.uint16 or indices.shape[1:] != (2, 32):
        raise ValueError("indices must be uint16[N,2,32]")
    if sides.shape != (len(indices),) or not np.all((sides == 0) | (sides == 1)):
        raise ValueError("sides must contain one zero/one per feature row")
    if len(indices) and int(indices.max()) > PADDING_FEATURE:
        raise ValueError("feature index exceeds padding row")
    padded = np.zeros((NUM_FEATURES + 1, artifact.width), dtype=np.int16)
    padded[:NUM_FEATURES] = artifact.feature_weights
    result = np.empty(len(indices), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end].astype(np.int64, copy=False)
        accumulators = padded[batch_indices].sum(axis=2, dtype=np.int32)
        accumulators += artifact.feature_bias.astype(np.int32)[None, None, :]
        result[start:end] = quantized_predict_from_accumulators(
            artifact, accumulators, sides[start:end]
        )
    return result


@torch.no_grad()
def float_predict(
    loaded: LoadedFloatModel,
    indices: np.ndarray,
    sides: np.ndarray,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    if loaded.format_name != PRODUCTION_FLOAT_FORMAT or loaded.num_features != 768:
        raise ValueError("production Chess768 data require a production float checkpoint")
    result = np.empty(len(indices), dtype=np.float32)
    loaded.model.eval()
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        tensor_indices = torch.from_numpy(
            indices[start:end].astype(np.int64, copy=False)
        )
        tensor_sides = torch.from_numpy(sides[start:end].astype(np.int64, copy=False))
        result[start:end] = loaded.model(tensor_indices, tensor_sides).cpu().numpy()
    return result


def concatenate_datasets(
    paths: list[Path], *, legacy_mod768: bool, maximum_examples: int | None
) -> EncodedArrays:
    pieces: list[EncodedArrays] = []
    remaining = maximum_examples
    for path in paths:
        arrays = load_encoded_npz(
            path,
            legacy_chessbuckets_mod768=legacy_mod768,
            limit=remaining,
        )
        pieces.append(arrays)
        if remaining is not None:
            remaining -= arrays.count
            if remaining <= 0:
                break
    return EncodedArrays(
        indices=np.concatenate([item.indices for item in pieces]),
        sides=np.concatenate([item.sides for item in pieces]),
        targets=np.concatenate([item.targets for item in pieces]),
        quiet=None,
    )


def absolute_error_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "median_cp": float(np.percentile(values, 50)),
        "p95_cp": float(np.percentile(values, 95)),
        "p99_cp": float(np.percentile(values, 99)),
        "maximum_cp": float(np.max(values)),
        "mean_cp": float(np.mean(values)),
    }


def probability_loss_numpy(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred_prob = 1.0 / (1.0 + np.exp(-predictions.astype(np.float64) / 400.0))
    target_prob = 1.0 / (1.0 + np.exp(-targets.astype(np.float64) / 400.0))
    return float(np.mean(np.square(pred_prob - target_prob)))


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return null, not NaN, when constant/short samples have no correlation."""
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    if len(left64) < 2 or np.std(left64) == 0.0 or np.std(right64) == 0.0:
        return None
    return float(np.corrcoef(left64, right64)[0, 1])


def crosscheck_runtime(
    artifact: QuantizedArtifact, indices: np.ndarray, sides: np.ndarray
) -> dict[str, int]:
    """Check both orientations of sampled accumulators against the deployed tail."""
    from nnue_lab.production.inference_numba import evaluate_ready

    artifact.validate()
    sampled = np.linspace(0, len(indices) - 1, min(1024, len(indices)), dtype=np.int64)
    padded = np.zeros((NUM_FEATURES + 1, artifact.width), dtype=np.int16)
    padded[:NUM_FEATURES] = artifact.feature_weights
    accumulators = padded[indices[sampled]].sum(axis=2, dtype=np.int32)
    accumulators += artifact.feature_bias[None, None, :].astype(np.int32)
    checked = 0
    for side in (0, 1):
        sample_sides = np.full(len(sampled), side, dtype=np.uint8)
        exported = quantized_predict_from_accumulators(artifact, accumulators, sample_sides)
        for row, accumulator in enumerate(accumulators):
            reference = evaluate_quantized_numpy(
                accumulator, side, artifact, validate_model=False
            )
            deployed = int(
                evaluate_ready(
                    accumulator,
                    side,
                    artifact.output_weights,
                    artifact.output_bias,
                    artifact.qa,
                    artifact.qb,
                    artifact.evaluation_scale_cp,
                )
            )
            if reference != deployed or reference != int(exported[row]):
                raise AssertionError(f"Q1 runtime mismatch at row {sampled[row]}, side {side}")
            checked += 1
    np.testing.assert_array_equal(
        quantized_predict(artifact, indices[sampled], sides[sampled]),
        quantized_predict_from_accumulators(artifact, accumulators, sides[sampled]),
    )
    return {"sampled_positions": len(sampled), "tail_checks": checked, "failures": 0}


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    paths = [path.resolve() for path in args.crosscheck_data]
    reject_pristine_before_selection(paths, args.final_selection_locked)
    loaded = load_float_model(args.checkpoint.resolve())
    if loaded.format_name != PRODUCTION_FLOAT_FORMAT:
        raise ValueError("Q1 export accepts only production Chess768 checkpoints")
    state = loaded.model.state_dict()
    feature_weights = quantise_checked(
        state["feature_weights.weight"].detach().cpu().numpy(), args.qa, "feature_weights"
    )
    feature_bias = quantise_checked(
        state["feature_bias"].detach().cpu().numpy(), args.qa, "feature_bias"
    )
    output_weights = quantise_checked(
        state["output.weight"].detach().cpu().numpy().reshape(-1),
        args.qb,
        "output_weights",
    )
    output_bias_array = quantise_checked(
        state["output.bias"].detach().cpu().numpy(),
        args.qa * args.qb,
        "output_bias",
    )
    metadata = {
        "created_utc": utc_now(),
        "participant_trained": True,
        "architecture": loaded.metadata.get("architecture", {}),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "checkpoint_metadata": loaded.metadata,
        "quantization_derivation": (
            "FWq=round(QA*FW), FBq=round(QA*FB), "
            "A=clip(FBq+sum(FWq),0,QA), OWq=round(QB*OW), "
            f"OBq=round(QA*QB*OB), S={EVALUATION_SCALE_CP}, cp={Q1_INTEGER_FORMULA}"
        ),
        "rounding": (
            "NumPy round-to-nearest-even for parameter quantization; both integer "
            "tail divisions truncate towards zero (C/C++ signed division)"
        ),
    }
    artifact = QuantizedArtifact(
        feature_weights=feature_weights,
        feature_bias=feature_bias,
        output_weights=output_weights,
        output_bias=np.int16(output_bias_array[0]),
        qa=args.qa,
        qb=args.qb,
        evaluation_scale_cp=EVALUATION_SCALE_CP,
        metadata=metadata,
    )
    artifact.validate()
    save_quantized_npz(args.output, artifact)
    saved = load_quantized(args.output)

    arrays = concatenate_datasets(
        paths,
        legacy_mod768=args.legacy_chessbuckets_mod768,
        maximum_examples=args.maximum_crosscheck_examples,
    )
    if arrays.count < args.minimum_crosscheck_examples:
        raise RuntimeError(
            f"Q1 cross-check has {arrays.count} examples; requires "
            f"{args.minimum_crosscheck_examples}"
        )
    started = time.perf_counter()
    float_cp = float_predict(
        loaded, arrays.indices, arrays.sides, batch_size=args.batch_size
    )
    quantized_cp = quantized_predict(
        saved, arrays.indices, arrays.sides, batch_size=args.batch_size
    )
    runtime_crosscheck = crosscheck_runtime(saved, arrays.indices, arrays.sides)
    elapsed = time.perf_counter() - started
    quantization_error = np.abs(quantized_cp.astype(np.float64) - float_cp)
    float_teacher_error = np.abs(float_cp.astype(np.float64) - arrays.targets)
    quantized_teacher_error = np.abs(
        quantized_cp.astype(np.float64) - arrays.targets
    )
    report: dict[str, Any] = {
        "format": "deepblue-perspective-chess768-q1-report-v1",
        "created_utc": utc_now(),
        "q1_contract": metadata["quantization_derivation"],
        "qa": args.qa,
        "qb": args.qb,
        "output_bias_scale": args.qa * args.qb,
        "evaluation_scale_cp": EVALUATION_SCALE_CP,
        "width": loaded.width,
        "deploy_parameter_count": NUM_FEATURES * loaded.width
        + loaded.width
        + 2 * loaded.width
        + 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "quantized_model": str(args.output.resolve()),
        "quantized_model_sha256": sha256_file(args.output),
        "quantized_model_bytes": args.output.stat().st_size,
        "crosscheck_data": [
            fingerprint_dataset(
                path,
                legacy_chessbuckets_mod768=args.legacy_chessbuckets_mod768,
            ).__dict__
            for path in paths
        ],
        "crosscheck_count": arrays.count,
        "crosscheck_is_pristine": any("pristine" in str(path).lower() for path in paths),
        "architecture_selection_declared_locked": args.final_selection_locked,
        "float_teacher_probability_mse": probability_loss_numpy(
            float_cp, arrays.targets
        ),
        "quantized_teacher_probability_mse": probability_loss_numpy(
            quantized_cp, arrays.targets
        ),
        "float_teacher_mae_cp": float(float_teacher_error.mean()),
        "quantized_teacher_mae_cp": float(quantized_teacher_error.mean()),
        "quantization_absolute_error": absolute_error_distribution(
            quantization_error
        ),
        "quantization_signed_bias_cp": float(
            np.mean(quantized_cp.astype(np.float64) - float_cp)
        ),
        "float_quantized_pearson_correlation": pearson_correlation(float_cp, quantized_cp),
        "numpy_export_numba_exact_agreement": runtime_crosscheck,
        "crosscheck_seconds": elapsed,
        "crosscheck_examples_per_second": arrays.count / elapsed,
        "integer_ranges": {
            "feature_weight_min": int(feature_weights.min()),
            "feature_weight_max": int(feature_weights.max()),
            "feature_bias_min": int(feature_bias.min()),
            "feature_bias_max": int(feature_bias.max()),
            "output_weight_min": int(output_weights.min()),
            "output_weight_max": int(output_weights.max()),
            "output_bias": int(output_bias_array[0]),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crosscheck-data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--qa", type=int, default=QA_DEFAULT)
    parser.add_argument("--qb", type=int, default=QB_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--minimum-crosscheck-examples", type=int, default=10_000)
    parser.add_argument("--maximum-crosscheck-examples", type=int)
    parser.add_argument("--legacy-chessbuckets-mod768", action="store_true")
    parser.add_argument("--final-selection-locked", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    for path in (args.output, args.report):
        if not path.is_relative_to(LAB_DIR):
            parser.error("Q1 model and report outputs must stay inside nnue_lab")
    if args.output.suffix.lower() != ".npz":
        parser.error("Q1 output must end in .npz")
    positive = (
        args.qa,
        args.qb,
        args.batch_size,
        args.threads,
        args.minimum_crosscheck_examples,
    )
    if any(value <= 0 for value in positive):
        parser.error("scales, batch size, threads, and minimum count must be positive")
    if args.maximum_crosscheck_examples is not None and args.maximum_crosscheck_examples <= 0:
        parser.error("maximum cross-check count must be positive")
    return args


if __name__ == "__main__":
    run_export(parse_args())
