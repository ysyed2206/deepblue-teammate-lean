"""Correctness tests for the donor-compatible feature mapper.

No external test framework: the project venv (pyproject.toml, uv.lock -- both
outside this directory's write boundary) does not carry pytest, and adding a
dependency there is out of scope for this lane. Each ``test_*`` function is a
plain assert-based case; ``run_all`` below discovers and runs them.

Run with: .venv/Scripts/python.exe nnue_lab/teammate_pawnstar/tests/test_features.py
"""

from __future__ import annotations

import random
import sys
import traceback
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import features as f  # noqa: E402


class TestFailure(AssertionError):
    pass


def test_constants_match_donor_header() -> None:
    assert f.NUM_KING_BUCKETS == 8
    assert f.INPUT_SIZE == 768
    assert f.FEATURE_ROWS == 6144
    assert f.HIDDEN_SIZE == 1024
    assert f.RANK_FLIP == 0x38


def test_king_bucket_map_matches_donor_table() -> None:
    expected = (
        [0, 0, 1, 1, 2, 2, 3, 3] * 4
        + [4, 4, 5, 5, 6, 6, 7, 7] * 4
    )
    for square in range(64):
        assert f.king_bucket(square, f.WHITE) == expected[square], square


def test_black_perspective_bucket_uses_rank_flip() -> None:
    for square in range(64):
        assert f.king_bucket(square, f.BLACK) == f.king_bucket(square ^ 0x38, f.WHITE)


def test_feature_row_white_perspective_hand_computed() -> None:
    row = f.feature_row(colour=f.WHITE, piece_type=1, square=1, king_square=4, perspective=f.WHITE)
    assert row == 2 * 768 + 0 * 384 + 1 * 64 + 1 == 1601


def test_feature_row_black_perspective_hand_computed() -> None:
    row = f.feature_row(colour=f.WHITE, piece_type=1, square=1, king_square=60, perspective=f.BLACK)
    assert row == 2041


def test_feature_row_rejects_out_of_range_inputs() -> None:
    for kwargs in (
        {"colour": 2, "piece_type": 0, "square": 0, "king_square": 4, "perspective": f.WHITE},
        {"colour": f.WHITE, "piece_type": 6, "square": 0, "king_square": 4, "perspective": f.WHITE},
        {"colour": f.WHITE, "piece_type": 0, "square": 64, "king_square": 4, "perspective": f.WHITE},
    ):
        try:
            f.feature_row(**kwargs)
        except ValueError:
            continue
        raise TestFailure(f"expected ValueError for {kwargs}")


def test_starting_position_feature_counts() -> None:
    board = chess.Board()
    white_features = f.active_features(board, f.WHITE)
    black_features = f.active_features(board, f.BLACK)
    assert len(white_features) == 32
    assert len(black_features) == 32
    assert len(set(white_features)) == 32
    assert len(set(black_features)) == 32


def test_all_rows_in_range() -> None:
    board = chess.Board()
    for perspective in (f.WHITE, f.BLACK):
        for row in f.active_features(board, perspective):
            assert 0 <= row < f.FEATURE_ROWS


def test_encode_board_padding() -> None:
    board = chess.Board()
    encoded = f.encode_board(board)
    assert encoded.shape == (2, f.MAX_PIECES)
    assert encoded.dtype.name == "uint16"
    assert (encoded[:, 32:] == f.PADDING_FEATURE).all()
    assert (encoded[:, :32] < f.FEATURE_ROWS).all()


def _mirrored_colour_board(board: chess.Board) -> chess.Board:
    mirrored = chess.Board(None)
    for square, piece in board.piece_map().items():
        mirrored.set_piece_at(square ^ 56, chess.Piece(piece.piece_type, not piece.color))
    mirrored.turn = not board.turn
    return mirrored


def test_perspective_symmetry_on_random_positions() -> None:
    for seed in range(20):
        rng = random.Random(seed)
        board = chess.Board()
        for _ in range(rng.randint(0, 40)):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(rng.choice(legal))
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            continue
        mirrored = _mirrored_colour_board(board)
        original_white = set(f.active_features(board, f.WHITE))
        mirrored_black = set(f.active_features(mirrored, f.BLACK))
        assert original_white == mirrored_black, f"seed={seed}"


def test_promotion_piece_type_is_queen_not_pawn() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
    board.push(chess.Move.from_uci("a7a8q"))
    pieces = dict((sq, (c, pt)) for sq, c, pt in f.board_pieces(board))
    assert pieces[chess.A8] == (f.WHITE, 4)


def test_underpromotion_piece_type() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
    board.push(chess.Move.from_uci("a7a8n"))
    pieces = dict((sq, (c, pt)) for sq, c, pt in f.board_pieces(board))
    assert pieces[chess.A8] == (f.WHITE, 1)


def test_en_passant_removes_captured_pawn_no_phantom() -> None:
    board = chess.Board("4k3/8/8/8/pP6/8/8/4K3 b - b3 0 1")
    board.push(chess.Move.from_uci("a4b3"))
    squares = {sq for sq, _, _ in f.board_pieces(board)}
    assert chess.B4 not in squares
    assert chess.A4 not in squares
    assert chess.B3 in squares


def test_castling_places_king_and_rook_correctly() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push(chess.Move.from_uci("e1g1"))
    pieces = dict((sq, (c, pt)) for sq, c, pt in f.board_pieces(board))
    assert pieces[chess.G1] == (f.WHITE, 5)
    assert pieces[chess.F1] == (f.WHITE, 3)
    assert chess.E1 not in pieces
    assert chess.H1 not in pieces


def test_queenside_castling_places_king_and_rook_correctly() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    board.push(chess.Move.from_uci("e8c8"))
    pieces = dict((sq, (c, pt)) for sq, c, pt in f.board_pieces(board))
    assert pieces[chess.C8] == (f.BLACK, 5)
    assert pieces[chess.D8] == (f.BLACK, 3)
    assert chess.E8 not in pieces
    assert chess.A8 not in pieces


def test_missing_king_raises() -> None:
    board = chess.Board(None)
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    try:
        f.king_squares(board)
    except ValueError:
        return
    raise TestFailure("expected ValueError for a board missing the black king")


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
    raise SystemExit(run_all())
