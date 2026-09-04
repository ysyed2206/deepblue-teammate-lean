"""Bounded exactness and paired elapsed-time experiment for optional Q1 tails."""

# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

PRODUCTION_DIR = Path(__file__).resolve().parent
LAB_DIR = PRODUCTION_DIR.parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(PRODUCTION_DIR / ".numba_cache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numba  # noqa: E402
import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

from nnue_lab.production.benchmark import (  # noqa: E402
    MOVE_FIXTURES,
    distribution,
    find_move,
    synthetic_parameters,
)
from nnue_lab.production.dataset import load_encoded_npz  # noqa: E402
from nnue_lab.production.inference_numba import (  # noqa: E402
    evaluate_ready,
    incremental_make,
    incremental_unmake,
    refresh_all,
)
from nnue_lab.production.inference_optimized import (  # noqa: E402
    _evaluate_ready_int32_products,
    _evaluate_ready_int32_sum,
    select_tail_kernel,
    tail_bounds,
)
from nnue_lab.production.model import load_quantized_npz  # noqa: E402


@njit(cache=True, nogil=True)
def evaluate_bank(tail, bank, weights, bias, qa, qb, scale):
    scores = np.empty((len(bank), 2), dtype=np.int64)
    for row in range(len(bank)):
        for side in range(2):
            scores[row, side] = tail(bank[row], side, weights, bias, qa, qb, scale)
    return scores


@njit(cache=True, nogil=True)
def benchmark_bank(tail, bank, weights, bias, qa, qb, scale, iterations):
    checksum = np.int64(0)
    for index in range(iterations):
        checksum += tail(
            bank[index % len(bank)], index & 1, weights, bias, qa, qb, scale
        )
    return checksum


@njit(cache=True, nogil=True)
def benchmark_update(
    tail, move, old_side, weights, accumulators, output, bias, qa, qb, scale, iterations
):
    checksum = np.int64(0)
    for index in range(iterations):
        if index & 1:
            incremental_unmake(move, old_side, weights, accumulators)
            side = old_side
        else:
            incremental_make(move, old_side, weights, accumulators)
            side = 1 - old_side
        checksum += tail(accumulators, side, output, bias, qa, qb, scale)
    return checksum


def reference_scores(bank, model):
    result = np.empty((len(bank), 2), dtype=np.int64)
    weights = model.output_weights.astype(np.int64)
    for start in range(0, len(bank), 256):
        clipped = np.clip(bank[start : start + 256], 0, model.qa).astype(np.int64)
        squared = clipped * clipped
        for side in (0, 1):
            dot = np.sum(squared[:, side] * weights[: model.width], axis=1)
            dot += np.sum(squared[:, 1 - side] * weights[model.width :], axis=1)
            first = np.sign(dot) * (np.abs(dot) // model.qa) + int(model.output_bias)
            scaled = first * model.evaluation_scale_cp
            result[start : start + len(clipped), side] = np.sign(scaled) * (
                np.abs(scaled) // (model.qa * model.qb)
            )
    return result


def build_bank(indices, model, rng, count):
    bank = np.empty((count * 2, 2, model.width), dtype=np.int32)
    padded = np.zeros((769, model.width), dtype=np.int16)
    padded[:768] = model.feature_weights
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        bank[start:stop] = padded[indices[start:stop]].sum(axis=2, dtype=np.int32)
        bank[start:stop] += model.feature_bias
    bank[count:] = rng.integers(
        -2_147_483_648, 2_147_483_648, size=bank[count:].shape, dtype=np.int32
    )
    # Dense values around clipping boundaries supplement full-range randoms.
    bank[count : count + count // 2] = rng.integers(
        -20, model.qa + 21, size=bank[count : count + count // 2].shape, dtype=np.int32
    )
    for index, value in enumerate((-2_147_483_648, -1, 0, 1, 254, 255, 256, 2_147_483_647)):
        bank[count + index].fill(value)
    return bank


def check_model(model, indices, count, rng):
    bounds = tail_bounds(model)
    kernels = {"canonical": evaluate_ready}
    if bounds["int32_products_safe"]:
        kernels["int32_products_int64_sum"] = _evaluate_ready_int32_products
    if bounds["int32_reduction_safe"]:
        kernels["int32_sum"] = _evaluate_ready_int32_sum
    bank = build_bank(indices, model, rng, count)
    args = (
        model.output_weights,
        model.output_bias,
        model.qa,
        model.qb,
        model.evaluation_scale_cp,
    )
    expected = reference_scores(bank, model)
    checks = {}
    for name, kernel in kernels.items():
        actual = evaluate_bank(kernel, bank, *args)
        np.testing.assert_array_equal(actual, expected, err_msg=name)
        checks[name] = {"positions": len(bank), "tail_checks": actual.size, "failures": 0}
    return bank, kernels, args, {
        "bounds": bounds,
        "positions_from_real_feature_rows": count,
        "random_and_extreme_positions": count,
        "checks": checks,
        "selected_kernel": select_tail_kernel(model).__name__,
    }


def paired_benchmark(model, bank, kernels, args, iterations, trials, rng):
    bank = np.ascontiguousarray(bank[:128])
    _, mail, _, move, old_side = find_move(*MOVE_FIXTURES["quiet_update"])
    acc = np.empty((2, model.width), dtype=np.int32)
    refresh_all(mail, model.feature_weights, model.feature_bias, acc)
    original = acc.copy()
    calls = {}
    for name, kernel in kernels.items():
        calls[("tail_bank", name)] = lambda kernel=kernel: benchmark_bank(
            kernel, bank, *args, iterations
        )
        calls[("quiet_update_plus_tail", name)] = lambda kernel=kernel: benchmark_update(
            kernel, move, old_side, model.feature_weights, acc, *args, iterations
        )
    # All signatures compiled before elapsed measurements; checksum checked per trial.
    checksums = {key: int(call()) for key, call in calls.items()}
    np.testing.assert_array_equal(acc, original)
    samples = {key: [] for key in calls}
    keys = list(calls)
    for _ in range(trials):
        rng.shuffle(keys)
        for key in keys:
            started = time.perf_counter_ns()
            checksum = int(calls[key]())
            samples[key].append((time.perf_counter_ns() - started) / iterations)
            assert checksum == checksums[key]
    np.testing.assert_array_equal(acc, original)
    result = {}
    for scenario in ("tail_bank", "quiet_update_plus_tail"):
        same_checksums = {checksums[(scenario, name)] for name in kernels}
        assert len(same_checksums) == 1
        baseline = np.median(samples[(scenario, "canonical")])
        result[scenario] = {
            name: {
                **distribution(samples[(scenario, name)]),
                "samples_ns": samples[(scenario, name)],
                "median_speedup_over_canonical": float(
                    baseline / np.median(samples[(scenario, name)])
                ),
                "checksum": checksums[(scenario, name)],
            }
            for name in kernels
        }
    return result


def fallback_tests(model, rng):
    checks = []
    for qa, magnitude, expected in (
        (255, 32767, _evaluate_ready_int32_products),
        (256, 32767, evaluate_ready),
    ):
        weights = np.full(2 * model.width, magnitude, dtype=np.int16)
        weights[::2] = -32768
        modified = replace(model, output_weights=weights, qa=qa)
        selected = select_tail_kernel(modified)
        assert selected is expected
        bank = rng.integers(-1000, 1001, size=(1000, 2, model.width), dtype=np.int32)
        bank[0].fill(qa)
        bank[1].fill(0)
        args = (weights, modified.output_bias, qa, modified.qb, modified.evaluation_scale_cp)
        actual = evaluate_bank(selected, bank, *args)
        np.testing.assert_array_equal(actual, reference_scores(bank, modified))
        checks.append({
            "qa": qa,
            "output_range": [-32768, 32767],
            "selected": selected.__name__,
            "bounds": tail_bounds(modified),
            "tail_checks": actual.size,
            "failures": 0,
        })
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--h512-model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=11)
    args = parser.parse_args()
    if not args.report.resolve().is_relative_to(LAB_DIR):
        parser.error("report must stay in nnue_lab")
    if "pristine" in str(args.data).lower():
        parser.error("this runtime-only experiment does not read pristine data")
    if args.count < 10_000 or args.trials < 9 or args.iterations <= 0 or args.iterations & 1:
        parser.error("requires >=10000 count, >=9 trials, and positive even iterations")
    rng = np.random.default_rng(20260919)
    arrays = load_encoded_npz(args.data, limit=args.count)
    if arrays.count < args.count:
        parser.error("insufficient feature positions")
    h256 = load_quantized_npz(args.model)
    h512 = (
        load_quantized_npz(args.h512_model)
        if args.h512_model is not None
        else synthetic_parameters(512, 20260919)
    )
    report = {
        "format": "deepblue-q1-tail-exact-paired-benchmark-v1",
        "canonical_unchanged": True,
        "timing": "perf_counter_ns elapsed, warmed JIT, randomized interleaved kernel order",
        "load_caveat": (
            "Concurrent six-thread H512 training and data acquisition/preprocessing "
            "may affect timing; no affinity pinning."
        ),
        "bank_semantics": (
            "128 real feature positions; varied position and side prevents invariant-call hoisting"
        ),
        "update_semantics": (
            "alternating exact quiet make/unmake; restored arrays and matching checksums asserted"
        ),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "iterations_per_trial": args.iterations,
        "trials": args.trials,
        "models": [],
    }
    for model, path in ((h256, args.model), (h512, args.h512_model)):
        bank, kernels, params, correctness = check_model(model, arrays.indices, args.count, rng)
        measurements = paired_benchmark(
            model, bank, kernels, params, args.iterations, args.trials, rng
        )
        report["models"].append({
            "width": model.width,
            "model_kind": (
                "actual_artifact" if path is not None else "synthetic_weights_real_feature_rows"
            ),
            "model_path": str(path.resolve()) if path is not None else None,
            "model_sha256": (
                hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None
            ),
            "correctness": correctness,
            "fallback_tests": fallback_tests(model, rng),
            "measurements": measurements,
        })
        print(json.dumps(report["models"][-1], indent=2), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
