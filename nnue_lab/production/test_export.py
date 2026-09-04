"""Q1 serialization, signed arithmetic, and compressed FEN regressions."""

from __future__ import annotations

import gzip
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PRODUCTION_DIR = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(PRODUCTION_DIR / ".numba_cache"))

from nnue_lab.production.evaluate import error_metrics, load_fen_records  # noqa: E402
from nnue_lab.production.export import (  # noqa: E402
    Q1_FORMAT,
    load_quantized,
    pearson_correlation,
    quantized_predict,
    quantized_predict_from_accumulators,
    truncating_divide_array,
)
from nnue_lab.production.inference_numba import evaluate_ready  # noqa: E402
from nnue_lab.production.model import (  # noqa: E402
    Q1_INTEGER_FORMULA,
    QUANTIZED_FORMAT,
    QuantizedParameters,
    evaluate_quantized_numpy,
    load_quantized_npz,
    save_quantized_npz,
)


class ExportContractTests(unittest.TestCase):
    @staticmethod
    def parameters(width: int) -> QuantizedParameters:
        rng = np.random.default_rng(width)
        return QuantizedParameters(
            feature_weights=rng.integers(-15, 16, (768, width), dtype=np.int16),
            feature_bias=rng.integers(-30, 100, width, dtype=np.int16),
            output_weights=rng.integers(-40, 41, 2 * width, dtype=np.int16),
            output_bias=np.int16(-123),
            metadata={"checkpoint_sha256": "test-only", "architecture": {"width": width}},
        )

    def test_one_serialization_contract_preserves_metadata(self) -> None:
        self.assertEqual(Q1_FORMAT, QUANTIZED_FORMAT)
        for width in (256, 512):
            with self.subTest(width=width), TemporaryDirectory(dir=PRODUCTION_DIR) as directory:
                original = self.parameters(width)
                path = Path(directory) / "candidate_q1.npz"
                save_quantized_npz(path, original)
                for loaded in (load_quantized(path), load_quantized_npz(path)):
                    self.assertEqual(loaded.metadata, original.metadata)
                    self.assertEqual(loaded.width, width)
                    np.testing.assert_array_equal(loaded.output_weights, original.output_weights)
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(archive["integer_formula"].item(), Q1_INTEGER_FORMULA)
                    self.assertEqual(archive["version"].item(), 2)

    def test_declared_incompatible_formula_is_rejected(self) -> None:
        with TemporaryDirectory(dir=PRODUCTION_DIR) as directory:
            path = Path(directory) / "bad_q1.npz"
            save_quantized_npz(path, self.parameters(256))
            with np.load(path, allow_pickle=False) as archive:
                fields = {name: archive[name] for name in archive.files}
            fields["integer_formula"] = np.asarray("round_symmetric(single_division)")
            np.savez(path, **fields)
            with self.assertRaisesRegex(ValueError, "integer_formula mismatch"):
                load_quantized(path)

    def test_signed_truncation_and_first_division_are_not_rounding(self) -> None:
        values = np.asarray([-8, -7, -2, -1, 0, 1, 2, 7, 8], dtype=np.int64)
        np.testing.assert_array_equal(
            truncating_divide_array(values, 3), [-2, -2, 0, 0, 0, 0, 0, 2, 2]
        )
        accumulator = np.zeros((1, 2, 256), dtype=np.int32)
        accumulator[0, 0, 0] = 1
        weights = np.zeros(512, dtype=np.int16)
        for sign in (-1, 1):
            weights[0] = sign
            model = replace(
                self.parameters(256), output_weights=weights.copy(),
                output_bias=np.int16(sign), qa=3, qb=1,
            )
            expected = sign * 133
            self.assertEqual(evaluate_quantized_numpy(accumulator[0], 0, model), expected)
            self.assertEqual(
                int(quantized_predict_from_accumulators(model, accumulator, np.array([0]))[0]),
                expected,
            )

    def test_export_numpy_numba_agree_on_random_accumulators_both_widths(self) -> None:
        for width in (256, 512):
            with self.subTest(width=width):
                model = self.parameters(width)
                model.validate()
                rng = np.random.default_rng(2323 + width)
                accumulators = rng.integers(-1000, 1000, (1024, 2, width), dtype=np.int32)
                for side in (0, 1):
                    exported = quantized_predict_from_accumulators(
                        model, accumulators, np.full(1024, side, dtype=np.uint8)
                    )
                    for row, accumulator in enumerate(accumulators):
                        reference = evaluate_quantized_numpy(
                            accumulator, side, model, validate_model=False
                        )
                        self.assertEqual(int(exported[row]), reference)
                        self.assertEqual(
                            int(evaluate_ready(
                                accumulator, side, model.output_weights,
                                model.output_bias, model.qa, model.qb,
                                model.evaluation_scale_cp,
                            )),
                            reference,
                        )

    def test_vectorized_refresh_matches_tail_and_rejects_bad_side(self) -> None:
        model = self.parameters(256)
        indices = np.full((2, 2, 32), 768, dtype=np.uint16)
        indices[:, :, :2] = np.array([[[1, 33], [405, 420]], [[10, 50], [550, 751]]])
        sides = np.array([0, 1], dtype=np.uint8)
        padded = np.vstack((model.feature_weights, np.zeros((1, model.width), np.int16)))
        accumulators = padded[indices].sum(axis=2, dtype=np.int32)
        accumulators += model.feature_bias[None, None, :]
        np.testing.assert_array_equal(
            quantized_predict(model, indices, sides, batch_size=1),
            quantized_predict_from_accumulators(model, accumulators, sides),
        )
        with self.assertRaises(ValueError):
            quantized_predict(model, indices, np.array([0, 2]))


class EvaluationInputTests(unittest.TestCase):
    def test_plain_and_gzip_fen_records_load_identically(self) -> None:
        record = {"fen": "4k3/8/8/8/8/8/8/4K3 b - -", "target_cp": -25}
        with TemporaryDirectory(dir=PRODUCTION_DIR) as directory:
            plain = Path(directory) / "development_fens.jsonl"
            zipped = Path(directory) / "development_fens.jsonl.gz"
            payload = json.dumps(record) + "\n\n"
            plain.write_text(payload, encoding="utf-8")
            with gzip.open(zipped, "wt", encoding="utf-8") as handle:
                handle.write(payload)
            expected = load_fen_records(plain)
            self.assertEqual(load_fen_records(zipped), expected)
            self.assertTrue(expected[0]["fen"].endswith("0 1"))

    def test_signed_bias_and_undefined_correlation(self) -> None:
        predictions = np.array([11, 21, 31])
        targets = np.array([10, 20, 30])
        metrics = error_metrics(predictions, targets)
        self.assertEqual(metrics["signed_bias_cp"], 1.0)
        self.assertEqual(metrics["pearson_correlation"], 1.0)
        self.assertIsNone(pearson_correlation(np.ones(3), targets))


if __name__ == "__main__":
    unittest.main()
