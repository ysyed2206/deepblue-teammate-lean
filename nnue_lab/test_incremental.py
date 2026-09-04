"""50k exact incremental/full/unmake correctness gate against Deep Blue core."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

LAB_DIR = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(LAB_DIR / ".numba_cache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import chess  # noqa: E402
import numpy as np  # noqa: E402

from deepblue import fastcore as core  # noqa: E402
from nnue_lab.inference_numba import (  # noqa: E402
    bucket_of,
    evaluate_ready,
    incremental_after_move,
    incremental_unmake_move,
    refresh_all,
)

EXPLICIT_CASES = (
    ("quiet", chess.STARTING_FEN, "g1f3"),
    ("capture", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
    ("white_en_passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
    ("black_en_passant", "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3"),
    ("white_promotion", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
    ("white_underpromotion", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
    (
        "white_promotion_capture",
        "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "a7b8q",
    ),
    ("black_promotion", "4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1q"),
    (
        "black_promotion_capture",
        "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1",
        "a2b1q",
    ),
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
    ("white_rank_bucket", "4k3/8/8/8/4K3/8/8/8 w - - 0 1", "e4e5"),
    ("white_file_bucket", "7k/8/8/8/8/8/8/1K6 w - - 0 1", "b1c1"),
    ("black_rank_bucket", "8/8/8/4k3/8/8/8/4K3 b - - 0 1", "e5e4"),
    ("black_file_bucket", "1k6/8/8/8/8/8/8/7K b - - 0 1", "b8c8"),
)


class GateFailure(AssertionError):
    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__(json.dumps(details, indent=2))
        self.details = details


def load_model(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    loaded = np.load(path, allow_pickle=False)
    if str(loaded["format"].item()) != "deepblue-nnue-int16-v0":
        raise ValueError("unsupported model format")
    return (
        loaded["feature_weights"],
        loaded["feature_bias"],
        loaded["output_weights"],
        int(loaded["output_bias"].item()),
        int(loaded["qa"].item()),
        int(loaded["qb"].item()),
        int(loaded["evaluation_scale_cp"].item()),
    )


def find_move(
    bb: np.ndarray,
    occ: np.ndarray,
    mail: np.ndarray,
    st: np.ndarray,
    uci: str,
    stack: np.ndarray,
    pseudo: np.ndarray,
    undo: np.ndarray,
    ply: int,
) -> np.uint32:
    count = core.generate_legal(bb, occ, mail, st, stack, pseudo, undo, ply)
    for index in range(count):
        move = stack[ply, index]
        if core.move_to_uci(int(move)) == uci:
            return move
    raise ValueError(f"move {uci} not legal")


def classify(move: np.uint32) -> list[str]:
    from_square, to_square, piece, captured, promotion, is_ep, is_castle, _ = core.decode(
        int(move)
    )
    labels: list[str] = []
    if is_castle:
        labels.append("castling")
    if is_ep:
        labels.append("en_passant")
    if captured != core.NO_PIECE:
        labels.append("capture")
    if promotion != core.NO_PIECE:
        labels.append("promotion_capture" if captured != core.NO_PIECE else "promotion")
    if piece in (core.WK, core.BK):
        perspective = 0 if piece == core.WK else 1
        if bucket_of(from_square, perspective) != bucket_of(to_square, perspective):
            labels.append("king_bucket_refresh")
    if not labels:
        labels.append("quiet")
    return labels


def require_equal(
    condition: bool,
    *,
    stage: str,
    fen: str,
    uci: str,
    transition: int,
    extra: dict[str, Any] | None = None,
) -> None:
    if condition:
        return
    details: dict[str, Any] = {
        "stage": stage,
        "fen": fen,
        "move": uci,
        "transition": transition,
    }
    if extra:
        details.update(extra)
    raise GateFailure(details)


def check_transition(
    *,
    fen: str,
    move: np.uint32,
    transition: int,
    ply: int,
    bb: np.ndarray,
    occ: np.ndarray,
    mail: np.ndarray,
    st: np.ndarray,
    undo: np.ndarray,
    accumulators: np.ndarray,
    feature_weights: np.ndarray,
    feature_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: int,
    qa: int,
    qb: int,
    scale: int,
) -> None:
    uci = core.move_to_uci(int(move))
    old_side = int(st[0])
    board_before = (bb.copy(), occ.copy(), mail.copy(), st.copy())
    accumulator_before = accumulators.copy()
    evaluation_before = int(
        evaluate_ready(accumulators, old_side, output_weights, output_bias, qa, qb, scale)
    )

    core.make_move(bb, occ, mail, st, move, undo, ply)
    incremental_after_move(
        mail, move, old_side, feature_weights, feature_bias, accumulators
    )
    full_after = np.empty_like(accumulators)
    refresh_all(mail, feature_weights, feature_bias, full_after)
    require_equal(
        np.array_equal(accumulators, full_after),
        stage="after_make_accumulator",
        fen=fen,
        uci=uci,
        transition=transition,
        extra={
            "maximum_absolute_difference": int(
                np.max(np.abs(accumulators.astype(np.int64) - full_after.astype(np.int64)))
            )
        },
    )
    incremental_eval = int(
        evaluate_ready(accumulators, int(st[0]), output_weights, output_bias, qa, qb, scale)
    )
    full_eval = int(
        evaluate_ready(full_after, int(st[0]), output_weights, output_bias, qa, qb, scale)
    )
    require_equal(
        incremental_eval == full_eval,
        stage="after_make_evaluation",
        fen=fen,
        uci=uci,
        transition=transition,
        extra={"incremental_cp": incremental_eval, "full_cp": full_eval},
    )

    core.unmake_move(bb, occ, mail, st, move, undo, ply)
    incremental_unmake_move(
        mail, move, old_side, feature_weights, feature_bias, accumulators
    )
    require_equal(
        all(
            np.array_equal(current, expected)
            for current, expected in zip((bb, occ, mail, st), board_before, strict=True)
        ),
        stage="board_after_unmake",
        fen=fen,
        uci=uci,
        transition=transition,
    )
    require_equal(
        np.array_equal(accumulators, accumulator_before),
        stage="accumulator_after_unmake",
        fen=fen,
        uci=uci,
        transition=transition,
        extra={
            "maximum_absolute_difference": int(
                np.max(
                    np.abs(accumulators.astype(np.int64) - accumulator_before.astype(np.int64))
                )
            )
        },
    )
    restored_eval = int(
        evaluate_ready(accumulators, int(st[0]), output_weights, output_bias, qa, qb, scale)
    )
    require_equal(
        restored_eval == evaluation_before,
        stage="evaluation_after_unmake",
        fen=fen,
        uci=uci,
        transition=transition,
        extra={"restored_cp": restored_eval, "original_cp": evaluation_before},
    )


def load_fens(path: Path) -> list[str]:
    fens = [fen for _, fen, _ in EXPLICIT_CASES]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fens.append(str(json.loads(line)["fen"]))
    return fens


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    (
        feature_weights,
        feature_bias,
        output_weights,
        output_bias,
        qa,
        qb,
        scale,
    ) = load_model(args.model)
    width = feature_bias.shape[0]
    events: Counter[str] = Counter()
    transition = 0
    minimum_accumulator = 2**31 - 1
    maximum_accumulator = -(2**31)
    started = time.perf_counter()

    for label, fen, uci in EXPLICIT_CASES:
        bb, occ, mail, st = core.from_fen(fen)
        stack, pseudo, undo = core.new_search_buffers()
        accumulators = np.empty((2, width), dtype=np.int32)
        refresh_all(mail, feature_weights, feature_bias, accumulators)
        move = find_move(bb, occ, mail, st, uci, stack, pseudo, undo, 0)
        check_transition(
            fen=fen,
            move=move,
            transition=transition,
            ply=0,
            bb=bb,
            occ=occ,
            mail=mail,
            st=st,
            undo=undo,
            accumulators=accumulators,
            feature_weights=feature_weights,
            feature_bias=feature_bias,
            output_weights=output_weights,
            output_bias=output_bias,
            qa=qa,
            qb=qb,
            scale=scale,
        )
        events[label] += 1
        events.update(classify(move))
        transition += 1

    rng = random.Random(args.seed)
    roots = load_fens(args.fens)
    while transition < args.transitions:
        root_fen = rng.choice(roots)
        try:
            board = chess.Board(root_fen if len(root_fen.split()) == 6 else root_fen + " 0 1")
        except ValueError:
            continue
        if not board.is_valid():
            continue
        bb, occ, mail, st = core.from_fen(board.fen())
        stack, pseudo, undo = core.new_search_buffers()
        accumulators = np.empty((2, width), dtype=np.int32)
        if refresh_all(mail, feature_weights, feature_bias, accumulators) != 0:
            continue
        for ply in range(args.episode_plies):
            if transition >= args.transitions:
                break
            count = core.generate_legal(bb, occ, mail, st, stack, pseudo, undo, ply)
            if count == 0:
                break
            move = stack[ply, rng.randrange(count)]
            fen = board.fen()
            check_transition(
                fen=fen,
                move=move,
                transition=transition,
                ply=ply,
                bb=bb,
                occ=occ,
                mail=mail,
                st=st,
                undo=undo,
                accumulators=accumulators,
                feature_weights=feature_weights,
                feature_bias=feature_bias,
                output_weights=output_weights,
                output_bias=output_bias,
                qa=qa,
                qb=qb,
                scale=scale,
            )
            events.update(classify(move))
            minimum_accumulator = min(minimum_accumulator, int(accumulators.min()))
            maximum_accumulator = max(maximum_accumulator, int(accumulators.max()))
            old_side = int(st[0])
            core.make_move(bb, occ, mail, st, move, undo, ply)
            incremental_after_move(
                mail, move, old_side, feature_weights, feature_bias, accumulators
            )
            board.push_uci(core.move_to_uci(int(move)))
            transition += 1

    required_events = (
        "quiet",
        "capture",
        "en_passant",
        "promotion",
        "promotion_capture",
        "castling",
        "king_bucket_refresh",
    )
    missing = [event for event in required_events if events[event] == 0]
    if missing:
        raise GateFailure({"stage": "coverage", "missing_events": missing})
    return {
        "format": "deepblue-nnue-incremental-gate-v0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "model": str(args.model.resolve()),
        "width": width,
        "transitions": transition,
        "accumulator_mismatches": 0,
        "evaluation_mismatches": 0,
        "unmake_mismatches": 0,
        "failures": 0,
        "event_counts": dict(events),
        "minimum_accumulator_observed": minimum_accumulator,
        "maximum_accumulator_observed": maximum_accumulator,
        "elapsed_seconds": time.perf_counter() - started,
        "result": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fens", type=Path, required=True)
    parser.add_argument("--transitions", type=int, default=50_000)
    parser.add_argument("--episode-plies", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.transitions < 50_000:
        parser.error("the integration gate requires at least 50,000 transitions")
    if not args.report.resolve().is_relative_to(LAB_DIR):
        parser.error("report must stay inside nnue_lab")
    return args


def main() -> None:
    args = parse_args()
    try:
        report = run_gate(args)
    except GateFailure as error:
        report = {
            "format": "deepblue-nnue-incremental-gate-v0",
            "result": "FAIL",
            "failures": 1,
            "first_failure": error.details,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(1) from error
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
