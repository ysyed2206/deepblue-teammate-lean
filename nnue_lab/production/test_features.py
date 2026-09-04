"""Unit tests for the no-bucket feature and float-model contracts."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import chess
import numpy as np
import torch

from nnue_lab.production.features import (
    BLACK,
    NUM_FEATURES,
    PADDING_FEATURE,
    WHITE,
    active_feature_rows,
    encode_board,
    encode_mailbox,
    feature_index,
    mirror_colours_and_ranks,
)
from nnue_lab.production.model import (
    QA_DEFAULT,
    QB_DEFAULT,
    PerspectiveChess768NNUE,
    QuantizedParameters,
    evaluate_quantized_numpy,
    load_quantized_npz,
    quantise_int16,
    quantize_model,
    save_quantized_npz,
)

PRODUCTION_DIR = Path(__file__).resolve().parent


class FeatureTests(unittest.TestCase):
    def test_exact_feature_rows(self) -> None:
        # White pawn a2: own a2 for White, opposing a7 for Black.
        self.assertEqual(feature_index(0, chess.A2, WHITE), 8)
        self.assertEqual(feature_index(0, chess.A2, BLACK), 384 + 48)
        # Black knight h7: opposing h7 for White, own h2 for Black.
        self.assertEqual(feature_index(7, chess.H7, WHITE), 384 + 64 + 55)
        self.assertEqual(feature_index(7, chess.H7, BLACK), 64 + 15)

    def test_colour_rank_mirror_swaps_perspectives(self) -> None:
        boards = (
            chess.Board(),
            chess.Board(
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R "
                "w KQkq - 0 1"
            ),
            chess.Board("4k3/2p5/8/3Pp3/8/2N5/5P2/4K3 b - - 8 31"),
        )
        for board in boards:
            with self.subTest(fen=board.fen()):
                original = encode_board(board)
                mirrored = encode_board(mirror_colours_and_ranks(board))
                np.testing.assert_array_equal(
                    active_feature_rows(original, WHITE),
                    active_feature_rows(mirrored, BLACK),
                )
                np.testing.assert_array_equal(
                    active_feature_rows(original, BLACK),
                    active_feature_rows(mirrored, WHITE),
                )

    def test_mailbox_and_board_encoders_agree(self) -> None:
        from deepblue.fastcore import from_fen

        board = chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R "
            "w KQkq - 0 1"
        )
        _, _, mailbox, _ = from_fen(board.fen())
        np.testing.assert_array_equal(encode_board(board), encode_mailbox(mailbox))

    def test_padding_is_outside_vocabulary(self) -> None:
        encoded = encode_board(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
        self.assertEqual(NUM_FEATURES, 768)
        self.assertEqual(PADDING_FEATURE, NUM_FEATURES)
        self.assertEqual(int(np.count_nonzero(encoded == PADDING_FEATURE)), 60)

    def test_invalid_input_rejected(self) -> None:
        with self.assertRaises(ValueError):
            feature_index(12, 0, WHITE)
        mailbox = np.full(64, 15, dtype=np.int64)
        mailbox[3] = 14
        with self.assertRaises(ValueError):
            encode_mailbox(mailbox)


class FloatModelTests(unittest.TestCase):
    def test_widths_sizes_and_forward_shape(self) -> None:
        indices = torch.from_numpy(encode_board(chess.Board())).long().unsqueeze(0)
        sides = torch.tensor([WHITE], dtype=torch.long)
        for width in (256, 512):
            with self.subTest(width=width):
                model = PerspectiveChess768NNUE(width)
                self.assertEqual(model(indices, sides).shape, (1,))
                self.assertEqual(model.deploy_parameter_count(), 771 * width + 1)
                self.assertEqual(model.raw_int16_bytes(), 2 * (771 * width + 1))

    def test_float_evaluation_is_perspective_symmetric(self) -> None:
        torch.manual_seed(818)
        model = PerspectiveChess768NNUE(256).eval()
        board = chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R "
            "w KQkq - 0 1"
        )
        mirrored = mirror_colours_and_ranks(board)
        indices = torch.from_numpy(
            np.stack((encode_board(board), encode_board(mirrored)))
        ).long()
        sides = torch.tensor(
            [WHITE if board.turn else BLACK, WHITE if mirrored.turn else BLACK]
        )
        with torch.no_grad():
            predictions = model(indices, sides)
        self.assertAlmostEqual(float(predictions[0]), float(predictions[1]), places=5)

    def test_quantizer_rejects_overflow_and_quantizes_both_widths(self) -> None:
        with self.assertRaises(OverflowError):
            quantise_int16(np.asarray([200.0]), 255, "deliberate_overflow")
        for width in (256, 512):
            with self.subTest(width=width):
                quantized = quantize_model(PerspectiveChess768NNUE(width))
                quantized.validate()
                self.assertEqual(quantized.feature_weights.shape, (768, width))
                self.assertEqual(quantized.output_weights.shape, (2 * width,))

    def test_q1_bias_scale_and_exact_dequantisation(self) -> None:
        model = PerspectiveChess768NNUE(256).eval()
        with torch.no_grad():
            model.feature_weights.weight.zero_()
            model.feature_bias.zero_()
            model.output.weight.zero_()
            model.output.bias.fill_(0.5)
        quantized = quantize_model(model)
        self.assertEqual(
            int(quantized.output_bias), QA_DEFAULT * QB_DEFAULT // 2
        )
        accumulators = np.zeros((2, 256), dtype=np.int32)
        self.assertEqual(evaluate_quantized_numpy(accumulators, WHITE, quantized), 200)

        output_weights = np.zeros(512, dtype=np.int16)
        output_weights[0] = np.int16(QB_DEFAULT)
        exact = QuantizedParameters(
            feature_weights=np.zeros((768, 256), dtype=np.int16),
            feature_bias=np.zeros(256, dtype=np.int16),
            output_weights=output_weights,
            output_bias=np.int16(QA_DEFAULT * QB_DEFAULT // 2),
        )
        accumulators[WHITE, 0] = QA_DEFAULT
        self.assertEqual(evaluate_quantized_numpy(accumulators, WHITE, exact), 600)

    def test_quantized_artifact_round_trip(self) -> None:
        torch.manual_seed(919)
        expected = quantize_model(PerspectiveChess768NNUE(256))
        with TemporaryDirectory(dir=PRODUCTION_DIR) as temporary:
            path = Path(temporary) / "model_q1.npz"
            save_quantized_npz(path, expected)
            actual = load_quantized_npz(path)
        self.assertEqual(actual.width, expected.width)
        self.assertEqual(actual.qa, expected.qa)
        self.assertEqual(actual.qb, expected.qb)
        self.assertEqual(actual.evaluation_scale_cp, expected.evaluation_scale_cp)
        self.assertEqual(actual.output_bias, expected.output_bias)
        np.testing.assert_array_equal(actual.feature_weights, expected.feature_weights)
        np.testing.assert_array_equal(actual.feature_bias, expected.feature_bias)
        np.testing.assert_array_equal(actual.output_weights, expected.output_weights)


if __name__ == "__main__":
    unittest.main()
