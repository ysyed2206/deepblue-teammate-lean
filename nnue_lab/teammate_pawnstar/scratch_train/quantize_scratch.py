"""Quantize a genuinely from-scratch-trained (random initialization, no
donor weights anywhere in the lineage) ``nnue_lab.model.PerspectiveNNUE``
checkpoint into the ``PSN1``-stamped binary format ``deepblue/nnue.py``
loads directly. Width-agnostic (reads the checkpoint's own width) --
mirrors ``finetune/quantize_finetuned.py``'s math exactly, just without any
donor-derived starting point. Compliance note: QA/QB/SCALE are Pawnstar's
published *quantization scale constants* (a numeric convention), not their
trained weights -- reusing the convention is unrelated to the "weights must
come from training you started" rule this lane is built around.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # nnue_lab/teammate_pawnstar
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

import weights as donor_weights  # noqa: E402  (constants only: QA/QB/SCALE/NET_MAGIC/INPUT_SIZE/NUM_KING_BUCKETS)
from deepblue import nnue as dnnue  # noqa: E402  (width-flexible loader, for the round-trip check)
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
        output_weight_flat = model.output.weight.numpy().reshape(-1)
        stm_weights = checked_round_int16(output_weight_flat[:width] * QB, "output_weights[stm]")
        ntm_weights = checked_round_int16(output_weight_flat[width:] * QB, "output_weights[ntm]")
        output_bias = checked_round_int16(model.output.bias.numpy() * QA * QB, "output_bias")

    header = struct.pack(
        "<4sHHHHhhh14s",
        donor_weights.NET_MAGIC,
        donor_weights.NET_FORMAT_VERSION,
        donor_weights.INPUT_SIZE,
        donor_weights.NUM_KING_BUCKETS,
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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    model = PerspectiveNNUE(int(metadata["width"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    blob = quantize(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print(f"wrote {args.out} ({len(blob)} bytes), width={model.width}")

    fw, fb, w_stm, w_ntm, ob = dnnue.load_weights(args.out)
    print(f"round-trip check OK: feature_weights shape={fw.shape}, dtype={fw.dtype}")


if __name__ == "__main__":
    main()
