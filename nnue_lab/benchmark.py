"""Measure full refresh, incremental events, ready tail, and accepted HCE cost."""

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

LAB_DIR = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(LAB_DIR / ".numba_cache"))
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
from deepblue.fastsearch import evaluate as hce_evaluate  # noqa: E402
from nnue_lab.inference_numba import (  # noqa: E402
    benchmark_perspective_refresh_loop,
    benchmark_refresh_loop,
    benchmark_tail_loop,
    benchmark_update_pair_loop,
    refresh_all,
)

REPRESENTATIVE_FEN = (
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
)


@njit(cache=False, nogil=True)
def benchmark_hce_loop(bb, st, mg_table, eg_table, phase_table, iterations):
    checksum = np.int64(0)
    original_side = st[0]
    for index in range(iterations):
        st[0] = index & 1
        checksum += hce_evaluate(bb, st, mg_table, eg_table, phase_table)
    st[0] = original_side
    return checksum


def load_model(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=False)
    return {name: loaded[name] for name in loaded.files}


def find_move(fen: str, uci: str) -> tuple[tuple[np.ndarray, ...], np.uint32]:
    bb, occ, mail, st = core.from_fen(fen)
    stack, pseudo, undo = core.new_search_buffers()
    count = core.generate_legal(bb, occ, mail, st, stack, pseudo, undo, 0)
    for index in range(count):
        move = stack[0, index]
        if core.move_to_uci(int(move)) == uci:
            return (bb, occ, mail, st, undo), move
    raise ValueError(f"{uci} is not legal in {fen}")


def prepare_update(
    fen: str,
    uci: str,
    feature_weights: np.ndarray,
    feature_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.uint32, int, np.ndarray]:
    (bb, occ, mail, st, undo), move = find_move(fen, uci)
    pre_mail = mail.copy()
    old_side = int(st[0])
    accumulators = np.empty((2, feature_bias.shape[0]), dtype=np.int32)
    refresh_all(mail, feature_weights, feature_bias, accumulators)
    core.make_move(bb, occ, mail, st, move, undo, 0)
    return pre_mail, mail.copy(), move, old_side, accumulators


def measure(function: Callable[[], Any], operations: int, trials: int) -> dict[str, Any]:
    samples: list[float] = []
    checksums: list[int] = []
    for _ in range(trials):
        started = time.perf_counter_ns()
        checksum = function()
        elapsed = time.perf_counter_ns() - started
        samples.append(elapsed / operations)
        checksums.append(int(checksum))
    return {
        "median_ns": statistics.median(samples),
        "minimum_ns": min(samples),
        "maximum_ns": max(samples),
        "trials_ns": samples,
        "checksum": checksums[-1],
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    model = load_model(args.model)
    feature_weights = model["feature_weights"]
    feature_bias = model["feature_bias"]
    output_weights = model["output_weights"]
    output_bias = int(model["output_bias"].item())
    qa = int(model["qa"].item())
    qb = int(model["qb"].item())
    scale = int(model["evaluation_scale_cp"].item())
    width = int(model["width"].item())

    bb, _, mail, st = core.from_fen(REPRESENTATIVE_FEN)
    accumulators = np.empty((2, width), dtype=np.int32)
    refresh_all(mail, feature_weights, feature_bias, accumulators)
    ordinary = prepare_update(chess_start_fen(), "g1f3", feature_weights, feature_bias)
    capture = prepare_update(
        "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
        "e4d5",
        feature_weights,
        feature_bias,
    )
    bucket_refresh = prepare_update(
        "4k3/8/8/8/4K3/8/8/8 w - - 0 1",
        "e4e5",
        feature_weights,
        feature_bias,
    )

    # Compile every path before measurements.
    benchmark_refresh_loop(mail, feature_weights, feature_bias, accumulators, 1)
    benchmark_perspective_refresh_loop(
        mail, feature_weights, feature_bias, 0, accumulators[0], 1
    )
    benchmark_tail_loop(accumulators, 0, output_weights, output_bias, qa, qb, scale, 1)
    for scenario in (ordinary, capture, bucket_refresh):
        benchmark_update_pair_loop(
            *scenario[:4], feature_weights, feature_bias, scenario[4].copy(), 1
        )
    benchmark_hce_loop(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE, 1)

    refresh_accumulators = accumulators.copy()
    metrics: dict[str, Any] = {}
    metrics["full_refresh_two_perspectives"] = measure(
        lambda: benchmark_refresh_loop(
            mail,
            feature_weights,
            feature_bias,
            refresh_accumulators,
            args.refresh_iterations,
        ),
        args.refresh_iterations,
        args.trials,
    )
    metrics["one_perspective_refresh"] = measure(
        lambda: benchmark_perspective_refresh_loop(
            mail,
            feature_weights,
            feature_bias,
            0,
            refresh_accumulators[0],
            args.refresh_iterations,
        ),
        args.refresh_iterations,
        args.trials,
    )

    def update_measurement(scenario: tuple[Any, ...]) -> dict[str, Any]:
        scenario_accumulators = scenario[4].copy()
        # One loop does after-move plus unmake, so divide by 2*N updates.
        return measure(
            lambda: benchmark_update_pair_loop(
                *scenario[:4],
                feature_weights,
                feature_bias,
                scenario_accumulators,
                args.update_iterations,
            ),
            2 * args.update_iterations,
            args.trials,
        )

    metrics["ordinary_incremental_update"] = update_measurement(ordinary)
    metrics["capture_incremental_update"] = update_measurement(capture)
    metrics["king_bucket_update_including_one_refresh"] = update_measurement(bucket_refresh)
    metrics["evaluation_tail_ready_accumulators"] = measure(
        lambda: benchmark_tail_loop(
            accumulators,
            0,
            output_weights,
            output_bias,
            qa,
            qb,
            scale,
            args.tail_iterations,
        ),
        args.tail_iterations,
        args.trials,
    )
    metrics["accepted_hce_evaluation"] = measure(
        lambda: benchmark_hce_loop(
            bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE, args.hce_iterations
        ),
        args.hce_iterations,
        args.trials,
    )
    combined = (
        metrics["ordinary_incremental_update"]["median_ns"]
        + metrics["evaluation_tail_ready_accumulators"]["median_ns"]
    )
    report = {
        "format": "deepblue-nnue-numba-benchmark-v0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model.resolve()),
        "width": width,
        "trials": args.trials,
        "metrics": metrics,
        "key_incremental_plus_tail_median_ns": combined,
        "key_incremental_plus_tail_vs_hce_ratio": combined
        / metrics["accepted_hce_evaluation"]["median_ns"],
        "interpretation_note": (
            "full refresh is reported separately and is not the expected ordinary-node cost"
        ),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def chess_start_fen() -> str:
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--refresh-iterations", type=int, default=20_000)
    parser.add_argument("--update-iterations", type=int, default=200_000)
    parser.add_argument("--tail-iterations", type=int, default=500_000)
    parser.add_argument("--hce-iterations", type=int, default=500_000)
    args = parser.parse_args()
    if not args.report.resolve().is_relative_to(LAB_DIR):
        parser.error("benchmark report must stay inside nnue_lab")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
