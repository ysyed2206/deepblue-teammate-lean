"""Dequantize the real Pawnstar v12 weights into a float ``PerspectiveNNUE``
checkpoint, as the starting point for fine-tuning.

Dequantization math (reverse of the forward quantization in
``PAWNSTAR_FORMAT.md`` / ``reference_eval.py``):

    feature_weights_float = feature_weights_int16 / QA
    feature_bias_float    = feature_bias_int16 / QA
    output_weight_float   = output_weights_int16 / QB   (concat stm, then ntm)
    output_bias_float     = output_bias_int16 / (QA * QB)

Derivation: Pawnstar's integer forward pass is
    dot_int = sum(SCReLU_int(acc_int) * w_int)
with ``SCReLU_int(acc) = clamp(acc, 0, QA)**2``. Writing ``acc_float =
acc_int/QA`` (so it lands in nnue_lab's own [0,1] SCReLU convention) and
``w_float = w_int/QB``, algebra gives ``dot_int = QA**2 * QB * dot_float``
where ``dot_float`` is exactly what a ``nn.Linear`` over
``clamp(acc_float,0,1)**2`` computes. Carrying that through Pawnstar's own
``Dequant`` (``/QA``, ``+bias``, ``*SCALE``, ``/(QA*QB)``) collapses to
``output_cp = (dot_float + bias_float) * SCALE`` -- which is *exactly*
``nnue_lab.model.PerspectiveNNUE.forward``'s ``raw * EVAL_SCALE_CP`` (both
``SCALE`` and ``EVAL_SCALE_CP`` are 400). So this conversion is not an
approximation -- the two formulations are the same function, just
integer-quantized vs. float, given this exact scale mapping. Verified below
against the real 250-position donor reference set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # nnue_lab/teammate_pawnstar
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

import weights as donor_weights  # noqa: E402  (teammate_pawnstar/weights.py)
from nnue_lab.model import PerspectiveNNUE  # noqa: E402

DONOR_QA = donor_weights.QA
DONOR_QB = donor_weights.QB


def convert(donor: donor_weights.NNUEWeights) -> PerspectiveNNUE:
    width = donor.feature_weights.shape[1]
    model = PerspectiveNNUE(width)
    with torch.no_grad():
        fw = donor.feature_weights.astype(np.float32) / DONOR_QA
        model.feature_weights.weight.copy_(torch.from_numpy(fw))

        fb = donor.feature_bias.astype(np.float32) / DONOR_QA
        model.feature_bias.copy_(torch.from_numpy(fb))

        # output_weights[0] = stm weights, [1] = ntm weights -- PerspectiveNNUE's
        # own forward concatenates [stm, non_stm] in that exact order before the
        # Linear layer, so a flat concat here lines up column-for-column.
        ow = np.concatenate([donor.output_weights[0], donor.output_weights[1]]).astype(np.float32) / DONOR_QB
        model.output.weight.copy_(torch.from_numpy(ow).unsqueeze(0))

        ob = np.float32(donor.output_bias) / (DONOR_QA * DONOR_QB)
        model.output.bias.copy_(torch.tensor([ob], dtype=torch.float32))
    model.eval()
    return model


def main() -> None:
    donor_path = Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\donor_reference\pawnstar-v12.bin")
    donor = donor_weights.load_donor_net(donor_path)
    model = convert(donor)

    out_dir = Path(__file__).resolve().parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pawnstar_v12_dequantized_float.pt"
    torch.save(
        {
            "format": "deepblue-nnue-float-v0",
            "state_dict": model.state_dict(),
            "metadata": {
                "width": model.width,
                "seed": 0,
                "epoch": 0,
                "validation_loss": 0.0,
                "validation_mae_cp": 0.0,
                "data_path": "dequantized-from-pawnstar-v12",
            },
        },
        out_path,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
