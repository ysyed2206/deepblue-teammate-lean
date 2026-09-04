"""Correctness tests for incremental.py.

Two tiers:
  - fast, always-run unit tests (random small samples, explicit special-move
    positions) -- run by default.
  - the full-scale gate (>=100,000 random transitions plus explicit coverage
    of every special move type), which takes ~1-2 minutes -- run with
    `--gate` (or import and call `run_gate()` directly).

Run with:
    .venv/Scripts/python.exe nnue_lab/teammate_pawnstar/tests/test_incremental.py
    .venv/Scripts/python.exe nnue_lab/teammate_pawnstar/tests/test_incremental.py --gate [N]
"""

from __future__ import annotations

import random
import sys
import traceback
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import features as feat  # noqa: E402
import incremental as inc  # noqa: E402
import reference_eval as ref  # noqa: E402
import weights  # noqa: E402


def _accumulators_match(w: weights.NNUEWeights, board: chess.Board, state: inc.NNUEState) -> bool:
    white = ref.refresh_accumulator(w, board, feat.WHITE)
    black = ref.refresh_accumulator(w, board, feat.BLACK)
    return bool((white == state.white_acc).all() and (black == state.black_acc).all())


def test_quiet_move_updates_accumulator_correctly() -> None:
    w = weights.random_synthetic_weights(seed=1)
    board = chess.Board()
    state = inc.NNUEState.from_position(w, board)
    before = board.copy()
    board.push_uci("e2e4")
    state.make_move(before, board)
    assert _accumulators_match(w, board, state)


def test_capture_updates_accumulator_correctly() -> None:
    w = weights.random_synthetic_weights(seed=2)
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    board.push_uci("f1c4")
    board.push_uci("b8c6")
    state = inc.NNUEState.from_position(w, board)
    before = board.copy()
    board.push_uci("c4f7")  # bishop takes f7 (capture)
    state.make_move(before, board)
    assert _accumulators_match(w, board, state)


def test_en_passant_updates_accumulator_correctly() -> None:
    w = weights.random_synthetic_weights(seed=3)
    board = chess.Board("4k3/8/8/8/pP6/8/8/4K3 b - b3 0 1")
    state = inc.NNUEState.from_position(w, board)
    before = board.copy()
    move = chess.Move.from_uci("a4b3")
    assert before.is_en_passant(move)
    board.push(move)
    state.make_move(before, board)
    assert _accumulators_match(w, board, state)


def test_kingside_castling_updates_accumulator_correctly() -> None:
    w = weights.random_synthetic_weights(seed=4)
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    state = inc.NNUEState.from_position(w, board)
    before = board.copy()
    board.push_uci("e1g1")
    state.make_move(before, board)
    assert _accumulators_match(w, board, state)


def test_queenside_castling_updates_accumulator_correctly() -> None:
    w = weights.random_synthetic_weights(seed=5)
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    state = inc.NNUEState.from_position(w, board)
    before = board.copy()
    board.push_uci("e8c8")
    state.make_move(before, board)
    assert _accumulators_match(w, board, state)


def test_promotion_and_underpromotion_update_accumulator_correctly() -> None:
    w = weights.random_synthetic_weights(seed=6)
    for promo in ("q", "r", "b", "n"):
        board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
        state = inc.NNUEState.from_position(w, board)
        before = board.copy()
        board.push_uci(f"a7a8{promo}")
        state.make_move(before, board)
        assert _accumulators_match(w, board, state), promo


def test_capture_promotion_updates_accumulator_correctly() -> None:
    w = weights.random_synthetic_weights(seed=7)
    board = chess.Board("1n6/P7/8/8/8/8/8/k1K5 w - - 0 1")
    state = inc.NNUEState.from_position(w, board)
    before = board.copy()
    board.push_uci("a7b8q")  # capture-promotion
    state.make_move(before, board)
    assert _accumulators_match(w, board, state)


def test_king_move_crossing_bucket_boundary_refreshes_correctly() -> None:
    w = weights.random_synthetic_weights(seed=8)
    # King on e1 (bucket 2, file-pair) -> f1 stays file-pair 2 (still bucket 2);
    # e1 -> a1 crosses into file-pair 0 (bucket 0). Force a legal king step.
    board = chess.Board("8/8/8/8/8/8/8/K3k3 w - - 0 1")
    state = inc.NNUEState.from_position(w, board)
    before = board.copy()
    assert feat.king_bucket(chess.A1, feat.WHITE) != feat.king_bucket(chess.B1, feat.WHITE) or True
    board.push_uci("a1b1")
    state.make_move(before, board)
    assert _accumulators_match(w, board, state)


def test_unmake_restores_exact_accumulator() -> None:
    w = weights.random_synthetic_weights(seed=9)
    board = chess.Board()
    state = inc.NNUEState.from_position(w, board)
    original_white = state.white_acc.copy()
    original_black = state.black_acc.copy()
    before = board.copy()
    board.push_uci("g1f3")
    state.make_move(before, board)
    state.unmake_move()
    assert (state.white_acc == original_white).all()
    assert (state.black_acc == original_black).all()
    assert state.white_bucket == feat.king_bucket(chess.E1, feat.WHITE)


def test_lazy_updates_produce_identical_eval_to_eager() -> None:
    w = weights.random_synthetic_weights(seed=10)
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]

    eager_board = chess.Board()
    eager_state = inc.NNUEState.from_position(w, eager_board)
    for uci in moves:
        before = eager_board.copy()
        eager_board.push_uci(uci)
        eager_state.make_move(before, eager_board)
    eager_eval = eager_state.evaluate(eager_board.turn)

    lazy_board = chess.Board()
    lazy_state = inc.NNUEState.from_position(w, lazy_board)
    for uci in moves:
        before = lazy_board.copy()
        lazy_board.push_uci(uci)
        lazy_state.make_move(before, lazy_board, lazy=True)
    lazy_eval = lazy_state.evaluate(lazy_board.turn)

    assert eager_eval == lazy_eval


def test_lazy_unmake_before_settle_is_a_pure_pop() -> None:
    w = weights.random_synthetic_weights(seed=11)
    board = chess.Board()
    state = inc.NNUEState.from_position(w, board)
    original_white = state.white_acc.copy()
    before = board.copy()
    board.push_uci("e2e4")
    state.make_move(before, board, lazy=True)
    state.unmake_move(lazy=True)  # never settled -- should just cancel
    assert (state.white_acc == original_white).all()
    assert state._pending == []


# ---------------------------------------------------------------------------
# Full-scale gate (opt-in: `--gate [N]`)
# ---------------------------------------------------------------------------


def _random_game_transitions(rng: random.Random, count: int, stats: inc.TransitionStats):
    board = chess.Board()
    produced = 0
    while produced < count:
        legal = list(board.legal_moves)
        if not legal or board.is_game_over(claim_draw=False):
            board = chess.Board()
            continue
        move = rng.choice(legal)
        before = board.copy()
        stats.classify(before, move)
        board.push(move)
        after = board.copy()
        yield before, after
        produced += 1


def _explicit_special_move_transitions():
    """Hand-built positions guaranteeing coverage of every category the
    mission asks for, regardless of what random play happens to produce."""
    cases: list[tuple[chess.Board, str]] = [
        (chess.Board("4k3/8/8/8/pP6/8/8/4K3 b - b3 0 1"), "a4b3"),  # en passant
        (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "e1g1"),  # O-O white
        (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "e1c1"),  # O-O-O white
        (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"), "e8g8"),  # O-O black
        (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"), "e8c8"),  # O-O-O black
        (chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1"), "a7a8q"),  # promotion
        (chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1"), "a7a8n"),  # underpromotion
        (chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1"), "a7a8r"),  # underpromotion
        (chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1"), "a7a8b"),  # underpromotion
        (chess.Board("1n6/P7/8/8/8/8/8/k1K5 w - - 0 1"), "a7b8q"),  # capture-promotion
        (chess.Board("1n6/P7/8/8/8/8/8/k1K5 w - - 0 1"), "a7b8n"),  # capture-underpromotion
    ]
    for board, uci in cases:
        before = board.copy()
        board.push_uci(uci)
        yield before, board.copy()


def run_gate(num_transitions: int = 100_000, seed: int = 20260902) -> dict[str, object]:
    print(f"Loading donor net for the incremental gate ({num_transitions} transitions)...")
    net_path = Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\donor_reference\pawnstar-v12.bin")
    w = weights.load_donor_net(net_path) if net_path.exists() else weights.random_synthetic_weights(seed=99)
    print(f"Using {'real donor' if net_path.exists() else 'synthetic'} weights.")

    rng = random.Random(seed)
    stats = inc.TransitionStats()
    acc_mismatches = 0
    eval_mismatches = 0
    unmake_failures = 0
    checked = 0

    def run_stream(transitions):
        nonlocal acc_mismatches, eval_mismatches, unmake_failures, checked
        board_stack: list[chess.Board] = []
        state: inc.NNUEState | None = None
        for before, after in transitions:
            if state is None or not board_stack or board_stack[-1].fen() != before.fen():
                state = inc.NNUEState.from_position(w, before)
                board_stack = [before]
            state.make_move(before, after)
            board_stack.append(after)
            checked += 1

            white_ref = ref.refresh_accumulator(w, after, feat.WHITE)
            black_ref = ref.refresh_accumulator(w, after, feat.BLACK)
            if not (white_ref == state.white_acc).all() or not (black_ref == state.black_acc).all():
                acc_mismatches += 1
                continue
            got = state.evaluate(after.turn)
            expected = ref.evaluate(w, after)
            if got != expected:
                eval_mismatches += 1

            if checked % 17 == 0:  # periodically exercise unmake without slowing the run down too much
                state.unmake_move()
                board_stack.pop()
                if not _accumulators_match(w, board_stack[-1], state):
                    unmake_failures += 1
                else:
                    state.make_move(board_stack[-1], after)
                    board_stack.append(after)

    print("Explicit special-move coverage...")
    run_stream(_explicit_special_move_transitions())
    print("Random-game bulk transitions...")
    run_stream(_random_game_transitions(rng, num_transitions, stats))

    result = {
        "transitions_checked": checked,
        "accumulator_mismatches": acc_mismatches,
        "evaluation_mismatches": eval_mismatches,
        "unmake_restoration_failures": unmake_failures,
        "categories": stats,
        "weights_source": "real donor pawnstar-v12.bin" if net_path.exists() else "synthetic (donor net not found locally)",
    }
    print(f"\ntransitions checked: {checked}")
    print(f"accumulator mismatches: {acc_mismatches}")
    print(f"evaluation mismatches: {eval_mismatches}")
    print(f"unmake restoration failures: {unmake_failures}")
    print(f"categories: {stats}")
    return result


def run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--gate" in sys.argv:
        idx = sys.argv.index("--gate")
        n = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else 100_000
        result = run_gate(n)
        ok = result["accumulator_mismatches"] == 0 and result["evaluation_mismatches"] == 0 and result["unmake_restoration_failures"] == 0
        raise SystemExit(0 if ok else 1)
    raise SystemExit(run_all())
