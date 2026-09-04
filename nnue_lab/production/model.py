"""Float model and safe integer references for Perspective Chess768.

The production experiment deliberately varies only width: 256 or 512.  The
shared sparse transformer builds fixed white and black accumulators; SCReLU is
applied only in the scalar tail, ordered side-to-move first.

The Q1 deployment contract follows Pawnstar's exact scale ordering while the
implementation is independent: transformer parameters use ``QA`` units,
output weights use ``QB`` units, and the output bias uses ``QA * QB`` units.
The integer tail first divides the SCReLU dot product by ``QA``, adds the bias,
multiplies by the centipawn scale, and finally divides by ``QA * QB``.  Both
divisions truncate towards zero, matching C/C++ signed integer division.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor, nn

from nnue_lab.production.features import (
    MAX_PIECES,
    NUM_FEATURES,
    NUM_PERSPECTIVES,
    PADDING_FEATURE,
    encode_mailbox,
)

SUPPORTED_WIDTHS = (256, 512)
EVALUATION_SCALE_CP = 400
QA_DEFAULT = 255
QB_DEFAULT = 64
QUANTIZED_FORMAT: Final = "deepblue-perspective-chess768-q1-v2"
QUANTIZED_VERSION: Final = 2
Q1_INTEGER_FORMULA: Final = "trunc(S*(trunc(sum(OWq*A^2)/QA)+OBq)/(QA*QB))"


class PerspectiveChess768NNUE(nn.Module):
    """Shared Chess768 -> H, two SCReLU accumulators -> scalar STM score."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width not in SUPPORTED_WIDTHS:
            raise ValueError(f"width must be one of {SUPPORTED_WIDTHS}, got {width}")
        self.width = width
        self.feature_weights = nn.Embedding(NUM_FEATURES, width, sparse=True)
        self.feature_bias = nn.Parameter(torch.full((width,), 0.25))
        self.output = nn.Linear(2 * width, 1)
        nn.init.uniform_(self.feature_weights.weight, -0.02, 0.02)
        nn.init.normal_(self.output.weight, 0.0, 0.08 / math.sqrt(2 * width))
        nn.init.zeros_(self.output.bias)

    def accumulators(self, indices: Tensor) -> Tensor:
        """Build float accumulators from ``[batch, 2, 32]`` padded rows."""
        if indices.ndim != 3 or tuple(indices.shape[1:]) != (
            NUM_PERSPECTIVES,
            MAX_PIECES,
        ):
            raise ValueError(f"expected indices [batch,2,32], got {tuple(indices.shape)}")
        valid = indices < NUM_FEATURES
        safe_indices = indices.clamp_max(NUM_FEATURES - 1)
        weighted = self.feature_weights(safe_indices) * valid.unsqueeze(-1)
        return weighted.sum(dim=2) + self.feature_bias

    @staticmethod
    def screlu(accumulators: Tensor) -> Tensor:
        return torch.clamp(accumulators, 0.0, 1.0).square()

    def forward_from_accumulators(
        self, accumulators: Tensor, side_to_move: Tensor
    ) -> Tensor:
        """Evaluate already-built float accumulators in STM-first order."""
        if accumulators.ndim != 3 or tuple(accumulators.shape[1:]) != (
            NUM_PERSPECTIVES,
            self.width,
        ):
            raise ValueError(
                f"expected accumulators [batch,2,{self.width}], "
                f"got {tuple(accumulators.shape)}"
            )
        if side_to_move.ndim != 1 or side_to_move.shape[0] != accumulators.shape[0]:
            raise ValueError("side_to_move must have shape [batch]")
        activated = self.screlu(accumulators)
        batch = torch.arange(accumulators.shape[0], device=accumulators.device)
        stm = activated[batch, side_to_move]
        opponent = activated[batch, 1 - side_to_move]
        return self.output(torch.cat((stm, opponent), dim=1)).squeeze(1) * float(
            EVALUATION_SCALE_CP
        )

    def forward(self, indices: Tensor, side_to_move: Tensor) -> Tensor:
        return self.forward_from_accumulators(self.accumulators(indices), side_to_move)

    def deploy_parameter_count(self) -> int:
        return NUM_FEATURES * self.width + self.width + 2 * self.width + 1

    def raw_int16_bytes(self) -> int:
        return 2 * self.deploy_parameter_count()


def probability_mse(predicted_cp: Tensor, target_cp: Tensor) -> Tensor:
    """Stable bounded teacher loss retained from the proven baseline pipeline."""
    predicted = torch.sigmoid(predicted_cp / float(EVALUATION_SCALE_CP))
    target = torch.sigmoid(target_cp / float(EVALUATION_SCALE_CP))
    return torch.mean(torch.square(predicted - target))


def quantise_int16(array: np.ndarray, scale: int, name: str) -> np.ndarray:
    """Round a finite float array and reject any int16 overflow."""
    if scale <= 0:
        raise ValueError(f"{name} scale must be positive")
    source = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{name} contains non-finite values")
    rounded = np.rint(source * scale)
    limits = np.iinfo(np.int16)
    if rounded.size and (rounded.min() < limits.min or rounded.max() > limits.max):
        raise OverflowError(f"{name} does not fit int16 at scale {scale}")
    return rounded.astype(np.int16)


@dataclass(frozen=True)
class QuantizedParameters:
    """Validated arrays consumed directly by the integer Numba runtime."""

    feature_weights: np.ndarray
    feature_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.int16
    qa: int = QA_DEFAULT
    qb: int = QB_DEFAULT
    evaluation_scale_cp: int = EVALUATION_SCALE_CP
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.feature_bias.shape[0])

    def validate(self) -> None:
        width = self.width
        if width not in SUPPORTED_WIDTHS:
            raise ValueError(f"quantized width must be one of {SUPPORTED_WIDTHS}")
        expected_shapes = {
            "feature_weights": (NUM_FEATURES, width),
            "feature_bias": (width,),
            "output_weights": (2 * width,),
        }
        arrays = {
            "feature_weights": self.feature_weights,
            "feature_bias": self.feature_bias,
            "output_weights": self.output_weights,
        }
        for name, value in arrays.items():
            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} shape {value.shape} != {expected_shapes[name]}"
                )
            if value.dtype != np.int16:
                raise TypeError(f"{name} must be int16, got {value.dtype}")
            if not value.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
        if not isinstance(self.output_bias, np.int16):
            raise TypeError("output_bias must be an int16 scalar")
        if self.qa <= 0 or self.qb <= 0 or self.evaluation_scale_cp <= 0:
            raise ValueError("quantisation and evaluation scales must be positive")

        int32_max = int(np.iinfo(np.int32).max)
        transformer_bound = int(np.max(np.abs(self.feature_bias.astype(np.int64))))
        transformer_bound += MAX_PIECES * int(
            np.max(np.abs(self.feature_weights.astype(np.int64)))
        )
        if transformer_bound > int32_max:
            raise OverflowError("worst-case transformer sum exceeds int32")

        qa_squared = self.qa * self.qa
        dot_bound = qa_squared * sum(
            abs(int(value)) for value in self.output_weights
        )
        if dot_bound > int(np.iinfo(np.int64).max):
            raise OverflowError("worst-case SCReLU dot product exceeds int64")
        # The reference tail divides the dot by QA before adding a bias that is
        # already stored in QA*QB units.
        dequant_bound = dot_bound // self.qa + abs(int(self.output_bias))
        if dequant_bound * self.evaluation_scale_cp > int(np.iinfo(np.int64).max):
            raise OverflowError("worst-case scaled output exceeds int64")


def quantize_model(
    model: PerspectiveChess768NNUE,
    qa: int = QA_DEFAULT,
    qb: int = QB_DEFAULT,
    evaluation_scale_cp: int = EVALUATION_SCALE_CP,
) -> QuantizedParameters:
    """Convert a float model to the safe exact-int16 runtime contract."""
    state = model.state_dict()
    parameters = QuantizedParameters(
        feature_weights=np.ascontiguousarray(
            quantise_int16(
                state["feature_weights.weight"].detach().cpu().numpy(),
                qa,
                "feature_weights",
            )
        ),
        feature_bias=np.ascontiguousarray(
            quantise_int16(
                state["feature_bias"].detach().cpu().numpy(), qa, "feature_bias"
            )
        ),
        output_weights=np.ascontiguousarray(
            quantise_int16(
                state["output.weight"].detach().cpu().numpy().reshape(-1),
                qb,
                "output_weights",
            )
        ),
        output_bias=np.int16(
            quantise_int16(
                state["output.bias"].detach().cpu().numpy(),
                qa * qb,
                "output_bias",
            )[0]
        ),
        qa=qa,
        qb=qb,
        evaluation_scale_cp=evaluation_scale_cp,
    )
    parameters.validate()
    return parameters


def save_quantized_npz(path: str | Path, model: QuantizedParameters) -> None:
    """Write a self-describing, pickle-free Q1 artifact."""
    model.validate()
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("quantized artifact path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez(
        temporary,
        format=np.asarray(QUANTIZED_FORMAT),
        version=np.asarray(QUANTIZED_VERSION, dtype=np.int32),
        architecture=np.asarray("PerspectiveChess768-SCReLU"),
        num_features=np.asarray(NUM_FEATURES, dtype=np.int32),
        padding_feature=np.asarray(PADDING_FEATURE, dtype=np.int32),
        perspectives=np.asarray(NUM_PERSPECTIVES, dtype=np.int32),
        max_pieces=np.asarray(MAX_PIECES, dtype=np.int32),
        accumulator_order=np.asarray("white,black"),
        tail_order=np.asarray("side_to_move,non_side_to_move"),
        target_orientation=np.asarray("side_to_move_cp"),
        activation=np.asarray("SCReLU: clamp(accumulator,0,QA)^2"),
        feature_quantization=np.asarray("round(QA*value)"),
        output_weight_quantization=np.asarray("round(QB*value)"),
        output_bias_quantization=np.asarray("round(QA*QB*bias)"),
        integer_formula=np.asarray(Q1_INTEGER_FORMULA),
        feature_weights=model.feature_weights,
        feature_bias=model.feature_bias,
        output_weights=model.output_weights,
        output_bias=np.asarray(model.output_bias, dtype=np.int16),
        qa=np.asarray(model.qa, dtype=np.int32),
        qb=np.asarray(model.qb, dtype=np.int32),
        evaluation_scale_cp=np.asarray(model.evaluation_scale_cp, dtype=np.int32),
        width=np.asarray(model.width, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(model.metadata, sort_keys=True)),
    )
    os.replace(temporary, destination)


def _scalar_int(archive: np.lib.npyio.NpzFile, name: str) -> int:
    if name not in archive.files:
        raise ValueError(f"quantized artifact is missing {name!r}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"quantized artifact field {name!r} must be scalar")
    if value.dtype.kind not in "iu":
        raise TypeError(f"quantized artifact field {name!r} must be integer")
    return int(value.reshape(-1)[0])


def _int16_array(archive: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    value = np.asarray(archive[name])
    if value.dtype != np.int16:
        raise TypeError(f"quantized artifact field {name!r} must be int16")
    return np.ascontiguousarray(value)


def load_quantized_npz(path: str | Path) -> QuantizedParameters:
    """Load and validate a self-describing Q1 artifact without pickle."""
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError("quantized artifact path must end in .npz")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "format",
            "feature_weights",
            "feature_bias",
            "output_weights",
            "output_bias",
            "qa",
            "qb",
            "evaluation_scale_cp",
            "width",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"quantized artifact is missing fields: {missing}")
        format_value = np.asarray(archive["format"])
        if format_value.size != 1 or str(format_value.reshape(-1)[0]) != QUANTIZED_FORMAT:
            raise ValueError("unsupported quantized artifact format")
        # Early v2 runtime artifacts had only tensor/scaling fields. Preserve
        # their compatibility while verifying every declared contract field.
        for name, expected in (
            ("version", QUANTIZED_VERSION),
            ("num_features", NUM_FEATURES),
            ("padding_feature", PADDING_FEATURE),
            ("perspectives", NUM_PERSPECTIVES),
            ("max_pieces", MAX_PIECES),
        ):
            if name in archive.files and _scalar_int(archive, name) != expected:
                raise ValueError(f"quantized artifact {name} mismatch")
        for name, expected in (
            ("accumulator_order", "white,black"),
            ("tail_order", "side_to_move,non_side_to_move"),
            ("target_orientation", "side_to_move_cp"),
            ("output_bias_quantization", "round(QA*QB*bias)"),
            ("integer_formula", Q1_INTEGER_FORMULA),
        ):
            if name in archive.files and str(archive[name].item()) != expected:
                raise ValueError(f"quantized artifact {name} mismatch")
        metadata = (
            json.loads(str(archive["metadata_json"].item()))
            if "metadata_json" in archive.files
            else {}
        )
        if not isinstance(metadata, dict):
            raise TypeError("quantized artifact metadata must be a JSON object")
        feature_weights = _int16_array(archive, "feature_weights")
        feature_bias = _int16_array(archive, "feature_bias")
        output_weights = _int16_array(archive, "output_weights")
        output_bias_raw = np.asarray(archive["output_bias"])
        if output_bias_raw.size != 1:
            raise ValueError("quantized artifact output_bias must be scalar")
        if output_bias_raw.dtype != np.int16:
            raise TypeError("quantized artifact output_bias must be int16")
        parameters = QuantizedParameters(
            feature_weights=feature_weights,
            feature_bias=feature_bias,
            output_weights=output_weights,
            output_bias=np.int16(output_bias_raw.reshape(-1)[0]),
            qa=_scalar_int(archive, "qa"),
            qb=_scalar_int(archive, "qb"),
            evaluation_scale_cp=_scalar_int(archive, "evaluation_scale_cp"),
            metadata=metadata,
        )
        declared_width = _scalar_int(archive, "width")
    parameters.validate()
    if declared_width != parameters.width:
        raise ValueError(
            f"declared width {declared_width} != tensor width {parameters.width}"
        )
    return parameters


def refresh_quantized_numpy(
    mail: np.ndarray,
    model: QuantizedParameters,
    *,
    validate_model: bool = True,
) -> np.ndarray:
    """Trusted NumPy full-refresh reference returning ``int32[2, H]``.

    Hot differential gates may set ``validate_model=False`` after validating
    the immutable experiment artifact once at the gate boundary.
    """
    if validate_model:
        model.validate()
    encoded = encode_mailbox(mail)
    padded = np.zeros((NUM_FEATURES + 1, model.width), dtype=np.int16)
    padded[:NUM_FEATURES] = model.feature_weights
    accumulators = padded[encoded.astype(np.int64)].sum(axis=1, dtype=np.int32)
    accumulators += model.feature_bias.astype(np.int32)[None, :]
    return np.ascontiguousarray(accumulators)


def truncating_divide(numerator: int, denominator: int) -> int:
    """Integer division truncating towards zero without a float conversion."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


def evaluate_quantized_numpy(
    accumulators: np.ndarray,
    side_to_move: int,
    model: QuantizedParameters,
    *,
    validate_model: bool = True,
) -> int:
    """Trusted integer tail reference with conservative overflow validation."""
    if validate_model:
        model.validate()
    if accumulators.shape != (NUM_PERSPECTIVES, model.width):
        raise ValueError(
            f"accumulators must have shape (2, {model.width}), got {accumulators.shape}"
        )
    if accumulators.dtype != np.int32:
        raise TypeError(f"accumulators must be int32, got {accumulators.dtype}")
    if side_to_move not in (0, 1):
        raise ValueError("side_to_move must be zero or one")
    clipped = np.clip(accumulators, 0, model.qa).astype(np.int64)
    activated = clipped * clipped
    dot = int(activated[side_to_move] @ model.output_weights[: model.width])
    dot += int(
        activated[1 - side_to_move]
        @ model.output_weights[model.width :]
    )
    dequantized = truncating_divide(dot, model.qa) + int(model.output_bias)
    return truncating_divide(
        dequantized * model.evaluation_scale_cp, model.qa * model.qb
    )
