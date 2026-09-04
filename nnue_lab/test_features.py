"""Feature indexing and perspective-symmetry gates."""

from __future__ import annotations

import unittest

import chess
import numpy as np

from nnue_lab.features import (
    BLACK,
    NUM_FEATURES,
    PADDING_FEATURE,
    WHITE,
    encode_board,
    encode_mailbox,
    feature_index,
    king_bucket,
    mirrored_colour_board,
    normalize_fen,
    orient_square,
)


class FeatureTests(unittest.TestCase):
    def test_orientation_is_vertical_only(self) -> None:
        for square in range(64):
            self.assertEqual(orient_square(square, WHITE), square)
            self.assertEqual(orient_square(square, BLACK), square ^ 56)
            self.assertEqual(orient_square(square, BLACK) & 7, square & 7)

    def test_all_king_buckets(self) -> None:
        for perspective in (WHITE, BLACK):
            seen: set[int] = set()
            for square in range(64):
                bucket = king_bucket(square, perspective)
                self.assertGreaterEqual(bucket, 0)
                self.assertLess(bucket, 8)
                seen.add(bucket)
            self.assertEqual(seen, set(range(8)))

    def test_feature_vocabulary_bounds_and_uniqueness(self) -> None:
        rows: set[int] = set()
        # Use one representative king square for each perspective bucket.
        representatives = (0, 2, 4, 6, 32, 34, 36, 38)
        for perspective in (WHITE, BLACK):
            king_squares = (
                representatives
                if perspective == WHITE
                else tuple(square ^ 56 for square in representatives)
            )
            for king_square in king_squares:
                for piece in range(12):
                    for square in range(64):
                        row = feature_index(piece, square, king_square, perspective)
                        self.assertGreaterEqual(row, 0)
                        self.assertLess(row, NUM_FEATURES)
                        rows.add(row)
        self.assertEqual(len(rows), NUM_FEATURES)

    def test_known_rows(self) -> None:
        # White pawn a2, White perspective king e1: bucket 2, own pawn.
        self.assertEqual(feature_index(0, chess.A2, chess.E1, WHITE), 2 * 768 + chess.A2)
        # The same absolute piece is an opponent pawn from Black's perspective.
        expected = 2 * 768 + 384 + (chess.A2 ^ 56)
        self.assertEqual(feature_index(0, chess.A2, chess.E8, BLACK), expected)

    def test_colour_mirror_swaps_perspectives(self) -> None:
        fens = (
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        )
        for fen in fens:
            board = chess.Board(fen)
            mirrored = mirrored_colour_board(board)
            original_features = encode_board(board)
            mirrored_features = encode_board(mirrored)
            for perspective in (WHITE, BLACK):
                lhs = sorted(
                    int(value)
                    for value in original_features[perspective]
                    if value != PADDING_FEATURE
                )
                rhs = sorted(
                    int(value)
                    for value in mirrored_features[1 - perspective]
                    if value != PADDING_FEATURE
                )
                self.assertEqual(lhs, rhs)

    def test_mailbox_encoder_matches_python_chess(self) -> None:
        board = chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )
        mailbox = np.full(64, 15, dtype=np.int64)
        for square, piece in board.piece_map().items():
            mailbox[square] = (0 if piece.color else 6) + piece.piece_type - 1
        np.testing.assert_array_equal(encode_mailbox(mailbox), encode_board(board))

    def test_four_field_fen_normalization(self) -> None:
        short = "8/8/8/4k3/8/4K3/8/8 w - -"
        self.assertEqual(normalize_fen(short), short + " 0 1")


if __name__ == "__main__":
    unittest.main()

