"""Differential gate for the no-bucket Numba accumulator runtime."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

PRODUCTION_DIR = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(PRODUCTION_DIR / ".numba_cache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from deepblue import fastcore as core  # noqa: E402
from nnue_lab.production.inference_numba import (  # noqa: E402
    evaluate_ready,
    feature_row,
    incremental_make,
    incremental_unmake,
    refresh_all,
)
from nnue_lab.production.model import (  # noqa: E402
    QuantizedParameters,
    evaluate_quantized_numpy,
    load_quantized_npz,
    refresh_quantized_numpy,
)

NNUE_LAB_DIR = PRODUCTION_DIR.parent


def chess_start_fen() -> str:
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


SPECIAL_MOVES = (
    ("quiet", chess_start_fen(), "g1f3"),
    ("double_pawn_push", chess_start_fen(), "e2e4"),
    ("capture", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
    ("white_en_passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
    ("black_en_passant", "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3"),
    (
        "white_kingside_castle",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "e1g1",
    ),
    (
        "white_queenside_castle",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "e1c1",
    ),
    (
        "black_kingside_castle",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "e8g8",
    ),
    (
        "black_queenside_castle",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "e8c8",
    ),
    ("white_promotion", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
    ("black_promotion", "4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1n"),
    (
        "white_capture_promotion",
        "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "a7b8q",
    ),
    (
        "black_capture_promotion",
        "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1",
        "a2b1r",
    ),
    (
        "king_crosses_old_rank_bucket",
        "4k3/8/8/8/4K3/8/8/8 w - - 0 1",
        "e4e5",
    ),
    (
        "king_crosses_old_file_bucket",
        "7k/8/8/8/8/8/8/1K6 w - - 0 1",
        "b1c1",
    ),
    ("rook_changes_castling_rights", "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", "a1a2"),
)

RANDOM_ROOTS = (
    chess_start_fen(),
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
)


def synthetic_parameters(width: int, seed: int = 9901) -> QuantizedParameters:
    rng = np.random.default_rng(seed + width)
    parameters = QuantizedParameters(
        feature_weights=np.ascontiguousarray(
            rng.integers(-9, 10, size=(768, width), dtype=np.int16)
        ),
        feature_bias=np.ascontiguousarray(
            rng.integers(-24, 25, size=width, dtype=np.int16)
        ),
        output_weights=np.ascontiguousarray(
            rng.integers(-12, 13, size=2 * width, dtype=np.int16)
        ),
        output_bias=np.int16(3),
    )
    parameters.validate()
    return parameters


def find_move(
    bb: np.ndarray,
    occ: np.ndarray,
    mail: np.ndarray,
    state: np.ndarray,
    uci: str,
    stack: np.ndarray,
    pseudo: np.ndarray,
    undo: np.ndarray,
    ply: int,
) -> np.uint32:
    count = core.generate_legal(bb, occ, mail, state, stack, pseudo, undo, ply)
    for index in range(count):
        candidate = stack[ply, index]
        if core.move_to_uci(int(candidate)) == uci:
            return candidate
    raise AssertionError(f"expected legal move {uci}")


def classify(move: np.uint32) -> list[str]:
    _, _, piece, captured, promotion, is_ep, is_castle, is_double = core.decode(
        int(move)
    )
    labels: list[str] = []
    if is_castle:
        labels.append("castle")
    if is_ep:
        labels.append("en_passant")
    if is_double:
        labels.append("double_pawn_push")
    if captured != core.NO_PIECE:
        labels.append("capture")
    if promotion != core.NO_PIECE:
        labels.append("capture_promotion" if captured != core.NO_PIECE else "promotion")
    if piece in (core.WK, core.BK):
        labels.append("king_move_without_refresh")
    if piece in (core.WR, core.BR):
        labels.append("rook_move")
    if not labels:
        labels.append("quiet")
    return labels


def verify_transition(
    *,
    move: np.uint32,
    ply: int,
    bb: np.ndarray,
    occ: np.ndarray,
    mail: np.ndarray,
    state: np.ndarray,
    undo: np.ndarray,
    accumulators: np.ndarray,
    model: QuantizedParameters,
    context: str,
) -> None:
    board_before = tuple(array.copy() for array in (bb, occ, mail, state))
    accumulator_before = accumulators.copy()
    old_side = int(state[0])
    before_reference = evaluate_quantized_numpy(
        accumulators, old_side, model, validate_model=False
    )
    before_numba = int(
        evaluate_ready(
            accumulators,
            old_side,
            model.output_weights,
            model.output_bias,
            model.qa,
            model.qb,
            model.evaluation_scale_cp,
        )
    )
    if before_numba != before_reference:
        raise AssertionError(f"{context}: NumPy/Numba tail mismatch before move")

    core.make_move(bb, occ, mail, state, move, undo, ply)
    incremental_make(move, old_side, model.feature_weights, accumulators)
    fully_refreshed = np.empty_like(accumulators)
    refresh_all(mail, model.feature_weights, model.feature_bias, fully_refreshed)
    if not np.array_equal(accumulators, fully_refreshed):
        difference = int(
            np.max(np.abs(accumulators.astype(np.int64) - fully_refreshed.astype(np.int64)))
        )
        raise AssertionError(f"{context}: incremental/full mismatch, maximum {difference}")
    numpy_full = refresh_quantized_numpy(mail, model, validate_model=False)
    if not np.array_equal(fully_refreshed, numpy_full):
        raise AssertionError(f"{context}: NumPy/Numba refresh mismatch")
    after_reference = evaluate_quantized_numpy(
        fully_refreshed, int(state[0]), model, validate_model=False
    )
    after_numba = int(
        evaluate_ready(
            accumulators,
            int(state[0]),
            model.output_weights,
            model.output_bias,
            model.qa,
            model.qb,
            model.evaluation_scale_cp,
        )
    )
    if after_numba != after_reference:
        raise AssertionError(f"{context}: NumPy/Numba tail mismatch after move")

    core.unmake_move(bb, occ, mail, state, move, undo, ply)
    incremental_unmake(move, old_side, model.feature_weights, accumulators)
    if not all(
        np.array_equal(actual, expected)
        for actual, expected in zip((bb, occ, mail, state), board_before, strict=True)
    ):
        raise AssertionError(f"{context}: Deep Blue board restoration mismatch")
    if not np.array_equal(accumulators, accumulator_before):
        raise AssertionError(f"{context}: accumulator restoration mismatch")
    restored = int(
        evaluate_ready(
            accumulators,
            int(state[0]),
            model.output_weights,
            model.output_bias,
            model.qa,
            model.qb,
            model.evaluation_scale_cp,
        )
    )
    if restored != before_reference:
        raise AssertionError(f"{context}: evaluation restoration mismatch")


def verify_king_delta_without_refresh(model: QuantizedParameters) -> None:
    fen = "4k3/8/8/8/4K3/8/8/8 w - - 0 1"
    bb, occ, mail, state = core.from_fen(fen)
    stack, pseudo, undo = core.new_search_buffers()
    move = find_move(bb, occ, mail, state, "e4e5", stack, pseudo, undo, 0)
    before = refresh_quantized_numpy(mail, model, validate_model=False)
    expected = before.copy()
    from_square, to_square, piece, _, _, _, _, _ = core.decode(int(move))
    for perspective in (0, 1):
        expected[perspective] += model.feature_weights[
            feature_row(piece, to_square, perspective)
        ].astype(np.int32)
        expected[perspective] -= model.feature_weights[
            feature_row(piece, from_square, perspective)
        ].astype(np.int32)
    incremental_make(move, int(state[0]), model.feature_weights, before)
    if not np.array_equal(before, expected):
        raise AssertionError("king move was not the exact ordinary two-row delta")


def _resolve_parameters(
    model_or_width: QuantizedParameters | int, seed: int
) -> tuple[QuantizedParameters, str]:
    if isinstance(model_or_width, QuantizedParameters):
        model_or_width.validate()
        return model_or_width, "artifact"
    return synthetic_parameters(model_or_width, seed), "synthetic"


def run_special_cases(
    model_or_width: QuantizedParameters | int,
) -> Counter[str]:
    model, _ = _resolve_parameters(model_or_width, 9901)
    width = model.width
    events: Counter[str] = Counter()
    for label, fen, uci in SPECIAL_MOVES:
        bb, occ, mail, state = core.from_fen(fen)
        stack, pseudo, undo = core.new_search_buffers()
        accumulators = np.empty((2, width), dtype=np.int32)
        refresh_all(mail, model.feature_weights, model.feature_bias, accumulators)
        move = find_move(bb, occ, mail, state, uci, stack, pseudo, undo, 0)
        verify_transition(
            move=move,
            ply=0,
            bb=bb,
            occ=occ,
            mail=mail,
            state=state,
            undo=undo,
            accumulators=accumulators,
            model=model,
            context=f"{label}:{fen}:{uci}",
        )
        events[label] += 1
        events.update(classify(move))
    verify_king_delta_without_refresh(model)
    required = {
        "quiet",
        "double_pawn_push",
        "capture",
        "en_passant",
        "castle",
        "promotion",
        "capture_promotion",
        "king_move_without_refresh",
        "rook_move",
    }
    missing = sorted(required.difference(events))
    if missing:
        raise AssertionError(f"special-move coverage missing: {missing}")
    return events


def run_random_gate(
    model_or_width: QuantizedParameters | int, transitions: int, seed: int
) -> dict[str, Any]:
    model, model_kind = _resolve_parameters(model_or_width, seed)
    width = model.width
    rng = random.Random(seed)
    completed = 0
    episodes = 0
    events: Counter[str] = Counter()
    started = time.perf_counter()
    while completed < transitions:
        root = RANDOM_ROOTS[rng.randrange(len(RANDOM_ROOTS))]
        bb, occ, mail, state = core.from_fen(root)
        root_board = tuple(array.copy() for array in (bb, occ, mail, state))
        stack, pseudo, undo = core.new_search_buffers()
        accumulators = refresh_quantized_numpy(mail, model, validate_model=False)
        root_accumulators = accumulators.copy()
        history: list[tuple[np.uint32, int]] = []
        for ply in range(min(96, transitions - completed)):
            count = core.generate_legal(bb, occ, mail, state, stack, pseudo, undo, ply)
            if count == 0:
                break
            move = stack[ply, rng.randrange(count)]
            context = (
                f"random transition {completed}, root {root}, ply {ply}, "
                f"move {core.move_to_uci(int(move))}"
            )
            verify_transition(
                move=move,
                ply=ply,
                bb=bb,
                occ=occ,
                mail=mail,
                state=state,
                undo=undo,
                accumulators=accumulators,
                model=model,
                context=context,
            )
            old_side = int(state[0])
            core.make_move(bb, occ, mail, state, move, undo, ply)
            incremental_make(move, old_side, model.feature_weights, accumulators)
            history.append((move, old_side))
            events.update(classify(move))
            completed += 1
            if completed >= transitions:
                break

        for ply in range(len(history) - 1, -1, -1):
            move, old_side = history[ply]
            core.unmake_move(bb, occ, mail, state, move, undo, ply)
            incremental_unmake(move, old_side, model.feature_weights, accumulators)
        if not all(
            np.array_equal(actual, expected)
            for actual, expected in zip((bb, occ, mail, state), root_board, strict=True)
        ):
            raise AssertionError(f"episode {episodes}: root board did not restore")
        if not np.array_equal(accumulators, root_accumulators):
            raise AssertionError(f"episode {episodes}: root accumulator did not restore")
        episodes += 1

    return {
        "result": "PASS",
        "format": "deepblue-perspective-chess768-incremental-gate-v2",
        "model_kind": model_kind,
        "width": width,
        "transitions": completed,
        "episodes": episodes,
        "accumulator_mismatches": 0,
        "evaluation_mismatches": 0,
        "unmake_mismatches": 0,
        "failures": 0,
        "events": dict(events),
        "elapsed_seconds": time.perf_counter() - started,
    }


class IncrementalTests(unittest.TestCase):
    def test_all_special_moves_h256(self) -> None:
        run_special_cases(256)

    def test_h512_shape_generic_smoke(self) -> None:
        model = synthetic_parameters(512)
        bb, occ, mail, state = core.from_fen(chess_start_fen())
        stack, pseudo, undo = core.new_search_buffers()
        accumulators = refresh_quantized_numpy(mail, model, validate_model=False)
        move = find_move(bb, occ, mail, state, "e2e4", stack, pseudo, undo, 0)
        verify_transition(
            move=move,
            ply=0,
            bb=bb,
            occ=occ,
            mail=mail,
            state=state,
            undo=undo,
            accumulators=accumulators,
            model=model,
            context="H512 smoke",
        )


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
    parser.add_argument("--transitions", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.transitions <= 0:
        parser.error("--transitions must be positive")
    return args


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
    events = run_special_cases(model_or_width)
    report = run_random_gate(model_or_width, args.transitions, args.seed)
    report["special_events"] = dict(events)
    report["model_path"] = model_path
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        write_report(args.report, report)
    print(rendered)


if __name__ == "__main__":
    main()
