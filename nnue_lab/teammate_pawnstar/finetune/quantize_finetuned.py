"""Quantize the fine-tuned float checkpoint back into a Pawnstar-``PSN1``-
stamped binary -- the exact same format ``weights.load_donor_net()`` reads,
so the fine-tuned network plugs directly into the already-tested
``reference_eval.py`` / ``incremental.py`` / ``numba_kernels.py`` pipeline
with zero further adaptation.

Quantization (the exact inverse of ``convert_donor_to_float.py``'s
dequantization, so round-tripping through float training and back is a
closed loop):

    feature_weights_int16 = round(feature_weights_float * QA)
    feature_bias_int16    = round(feature_bias_float * QA)
    output_weights_int16  = round(output_weight_float * QB)   (split back into stm/ntm halves)
    output_bias_int16     = round(output_bias_float * QA * QB)

All values are checked to fit int16 before casting (a silent overflow would
be a correctness bug, not a warning-worthy event).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # nnue_lab/teammate_pawnstar
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

import weights as donor_weights  # noqa: E402
from nnue_lab.model import PerspectiveNNUE  # noqa: E402

QA = donor_weights.QA
QB = donor_weights.QB
SCALE = donor_weights.SCALE


def checked_round_int16(array: np.ndarray, name: str) -> np.ndarray:
    rounded = np.rint(array.astype(np.float64))
    lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    if rounded.min() < lo or rounded.max() > hi:
        raise OverflowError(f"{name} does not fit int16: min={rounded.min()} max={rounded.max()}")
    return rounded.astype(np.int16)


def quantize(model: PerspectiveNNUE) -> bytes:
    width = model.width
    with torch.no_grad():
        feature_weights = checked_round_int16(model.feature_weights.weight.numpy() * QA, "feature_weights")
        feature_bias = checked_round_int16(model.feature_bias.numpy() * QA, "feature_bias")
        output_weight_flat = model.output.weight.numpy().reshape(-1)  # (2*width,)
        stm_weights = checked_round_int16(output_weight_flat[:width] * QB, "output_weights[stm]")
        ntm_weights = checked_round_int16(output_weight_flat[width:] * QB, "output_weights[ntm]")
        output_bias = checked_round_int16(model.output.bias.numpy() * QA * QB, "output_bias")

    header = struct.pack(
        "<4sHHHHhhh14s",
        donor_weights.NET_MAGIC,
        donor_weights.NET_FORMAT_VERSION,
        donor_weights.INPUT_SIZE,  # 768, fixed donor constant
        donor_weights.NUM_KING_BUCKETS,  # 8, fixed donor constant
        width,
        QA,
        QB,
        SCALE,
        b"\x00" * 14,
    )
    payload = (
        feature_weights.tobytes(order="C")
        + feature_bias.tobytes(order="C")
        + stm_weights.tobytes(order="C")
        + ntm_weights.tobytes(order="C")
        + output_bias.tobytes(order="C")
    )
    return header + payload


def main() -> None:
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / "finetuned_best.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    model = PerspectiveNNUE(int(metadata["width"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    blob = quantize(model)
    out_path = Path(__file__).resolve().parent / "checkpoints" / "finetuned_v1.bin"
    out_path.write_bytes(blob)
    print(f"wrote {out_path} ({len(blob)} bytes)")

    # Sanity check: the file our own loader parses back out with the expected header.
    loaded = donor_weights.load_donor_net(out_path)
    print(f"round-trip check: header={loaded.header}")
    print(f"feature_weights shape={loaded.feature_weights.shape}, dtype={loaded.feature_weights.dtype}")


if __name__ == "__main__":
    main()
