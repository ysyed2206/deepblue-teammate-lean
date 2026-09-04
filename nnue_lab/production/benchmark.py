"""Elapsed-time benchmark hooks for the isolated no-bucket Numba runtime."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PRODUCTION_DIR = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(PRODUCTION_DIR / ".numba_cache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

from deepblue import fastcore as core  # noqa: E402
from deepblue.fastsearch import (  # noqa: E402
    EG_TABLE,
    MG_TABLE,
    PHASE_TABLE,
)
from deepblue.fastsearch import (  # noqa: E402
    evaluate as hce_evaluate,
)
from nnue_lab.production.inference_numba import (  # noqa: E402
    benchmark_alternating_update_loop,
    benchmark_alternating_update_tail_loop,
    benchmark_refresh_loop,
    benchmark_tail_loop,
    refresh_all,
)
from nnue_lab.production.model import (  # noqa: E402
    QuantizedParameters,
    load_quantized_npz,
)

NNUE_LAB_DIR = PRODUCTION_DIR.parent

MOVE_FIXTURES = {
    "quiet_update": (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "g1f3",
    ),
    "double_pawn_push": (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "e2e4",
    ),
    "capture_update": ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
    "king_update": ("4k3/8/8/8/4K3/8/8/8 w - - 0 1", "e4e5"),
    "castle_update": (
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "e1g1",
    ),
    "promotion_update": ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
}


def synthetic_parameters(width: int, seed: int) -> QuantizedParameters:
    rng = np.random.default_rng(seed)
    parameters = QuantizedParameters(
        np.ascontiguousarray(
            rng.integers(-9, 10, size=(768, width), dtype=np.int16)
        ),
        np.ascontiguousarray(rng.integers(-24, 25, size=width, dtype=np.int16)),
        np.ascontiguousarray(
            rng.integers(-12, 13, size=2 * width, dtype=np.int16)
        ),
        np.int16(3),
    )
    parameters.validate()
    return parameters


def find_move(
    fen: str, uci: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.uint32, int]:
    bb, occ, mail, state = core.from_fen(fen)
    stack, pseudo, undo = core.new_search_buffers()
    count = core.generate_legal(bb, occ, mail, state, stack, pseudo, undo, 0)
    for index in range(count):
        move = stack[0, index]
        if core.move_to_uci(int(move)) == uci:
            return bb, mail, state, move, int(state[0])
    raise ValueError(f"move {uci} is not legal in benchmark fixture")


@njit(cache=True, nogil=True)
def benchmark_hce_loop(bb, state, mg_table, eg_table, phase_table, iterations):
    checksum = np.int64(0)
    original_side = state[0]
    for iteration in range(iterations):
        # Vary an evaluator input so LLVM cannot hoist an invariant call out of
        # the timing loop.  Even iteration counts restore the original side.
        state[0] = original_side if (iteration & 1) == 0 else 1 - original_side
        checksum += hce_evaluate(bb, state, mg_table, eg_table, phase_table)
    state[0] = original_side
    return checksum


def distribution(samples: list[float]) -> dict[str, float]:
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    return {
        "minimum_ns": float(ordered[0]),
        "p10_ns": float(np.percentile(ordered, 10)),
        "median_ns": float(statistics.median(samples)),
        "p90_ns": float(np.percentile(ordered, 90)),
        "maximum_ns": float(ordered[-1]),
    }


def measure(
    call: Callable[[], np.int64], operations: int, trials: int
) -> dict[str, float | int]:
    samples: list[float] = []
    checksum = 0
    for _ in range(trials):
        started = time.perf_counter_ns()
        checksum ^= int(call())
        elapsed = time.perf_counter_ns() - started
        samples.append(elapsed / operations)
    return {**distribution(samples), "trials": trials, "checksum": checksum}


def run_benchmark(
    model_or_width: QuantizedParameters | int,
    iterations: int,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0 or iterations & 1:
        raise ValueError("iterations must be a positive even number")
    if trials < 3:
        raise ValueError("at least three trials are required")
    if isinstance(model_or_width, QuantizedParameters):
        model = model_or_width
        model.validate()
        model_kind = "artifact"
    else:
        model = synthetic_parameters(model_or_width, seed)
        model_kind = "synthetic"
    width = model.width
    start_bb, start_mail, start_state, _, _ = find_move(
        *MOVE_FIXTURES["quiet_update"]
    )
    accumulators = np.empty((2, width), dtype=np.int32)
    refresh_all(start_mail, model.feature_weights, model.feature_bias, accumulators)

    # Compile every measured signature before starting the clock.
    benchmark_refresh_loop(
        start_mail, model.feature_weights, model.feature_bias, accumulators, 2
    )
    benchmark_tail_loop(
        accumulators,
        model.output_weights,
        model.output_bias,
        model.qa,
        model.qb,
        model.evaluation_scale_cp,
        2,
    )
    benchmark_hce_loop(start_bb, start_state, MG_TABLE, EG_TABLE, PHASE_TABLE, 2)
    measurements: dict[str, Any] = {}
    refresh_iterations = max(2, iterations // 10)
    if refresh_iterations & 1:
        refresh_iterations += 1
    measurements["full_refresh"] = measure(
        lambda: benchmark_refresh_loop(
            start_mail,
            model.feature_weights,
            model.feature_bias,
            accumulators,
            refresh_iterations,
        ),
        refresh_iterations,
        trials,
    )
    measurements["tail_only"] = measure(
        lambda: benchmark_tail_loop(
            accumulators,
            model.output_weights,
            model.output_bias,
            model.qa,
            model.qb,
            model.evaluation_scale_cp,
            iterations,
        ),
        iterations,
        trials,
    )
    measurements["hce_evaluation"] = measure(
        lambda: benchmark_hce_loop(
            start_bb,
            start_state,
            MG_TABLE,
            EG_TABLE,
            PHASE_TABLE,
            iterations,
        ),
        iterations,
        trials,
    )

    for name, fixture in MOVE_FIXTURES.items():
        _, mail, _, move, old_side = find_move(*fixture)
        local_accumulators = np.empty((2, width), dtype=np.int32)
        refresh_all(mail, model.feature_weights, model.feature_bias, local_accumulators)
        benchmark_alternating_update_loop(
            move, old_side, model.feature_weights, local_accumulators, 2
        )
        benchmark_alternating_update_tail_loop(
            move,
            old_side,
            model.feature_weights,
            local_accumulators,
            model.output_weights,
            model.output_bias,
            model.qa,
            model.qb,
            model.evaluation_scale_cp,
            2,
        )
        measurements[name] = measure(
            lambda move=move, old_side=old_side, acc=local_accumulators: (
                benchmark_alternating_update_loop(
                    move, old_side, model.feature_weights, acc, iterations
                )
            ),
            iterations,
            trials,
        )
        measurements[f"{name}_plus_tail"] = measure(
            lambda move=move, old_side=old_side, acc=local_accumulators: (
                benchmark_alternating_update_tail_loop(
                    move,
                    old_side,
                    model.feature_weights,
                    acc,
                    model.output_weights,
                    model.output_bias,
                    model.qa,
                    model.qb,
                    model.evaluation_scale_cp,
                    iterations,
                )
            ),
            iterations,
            trials,
        )

    return {
        "format": "deepblue-perspective-chess768-runtime-benchmark-v2",
        "model_kind": model_kind,
        "width": width,
        "iterations_per_trial": iterations,
        "trials": trials,
        "timing": "actual elapsed perf_counter_ns; JIT warm-up excluded",
        "hce_semantics": "alternating side-to-move prevents invariant-call hoisting",
        "update_semantics": (
            "alternating exact make/unmake deltas; each operation is one transition"
        ),
        "measurements": measurements,
        "side_to_move_fixture": int(start_state[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(256, 512), default=256)
    parser.add_argument(
        "--model",
        type=Path,
        help="self-describing Q1 .npz artifact; overrides --width synthetic data",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON report path under nnue_lab/",
    )
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def write_report(path: Path, report: dict[str, Any]) -> None:
    destination = path.resolve()
    try:
        destination.relative_to(NNUE_LAB_DIR.resolve())
    except ValueError as error:
        raise ValueError("--report must resolve inside nnue_lab/") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    model_or_width: QuantizedParameters | int
    if args.model is None:
        model_or_width = args.width
        model_path = None
    else:
        model_or_width = load_quantized_npz(args.model)
        model_path = str(args.model.resolve())
    report = run_benchmark(model_or_width, args.iterations, args.trials, args.seed)
    report["model_path"] = model_path
    if args.report is not None:
        write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
