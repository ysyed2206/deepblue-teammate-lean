"""Float SCReLU network and compact dataset helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from nnue_lab.features import NUM_FEATURES, PADDING_FEATURE

EVAL_SCALE_CP = 400.0


class PerspectiveNNUE(nn.Module):
    """6144 sparse rows -> shared H -> SCReLU -> 2H -> one scalar."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width
        # Sparse gradients keep CPU pilot cost proportional to active board
        # features instead of updating all 6144 rows after every batch.
        self.feature_weights = nn.Embedding(NUM_FEATURES, width, sparse=True)
        self.feature_bias = nn.Parameter(torch.full((width,), 0.25))
        self.output = nn.Linear(2 * width, 1)
        nn.init.uniform_(self.feature_weights.weight, -0.02, 0.02)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.08 / math.sqrt(2 * width))
        nn.init.zeros_(self.output.bias)

    def accumulators(self, indices: Tensor) -> Tensor:
        """Return float [batch,white/black,H] accumulators before SCReLU."""
        if indices.ndim != 3 or indices.shape[1:] != (2, 32):
            raise ValueError(f"expected indices [batch,2,32], got {tuple(indices.shape)}")
        valid = indices < NUM_FEATURES
        safe = indices.clamp_max(NUM_FEATURES - 1)
        gathered = self.feature_weights(safe)
        gathered = gathered * valid.unsqueeze(-1)
        return gathered.sum(dim=2) + self.feature_bias

    def forward(self, indices: Tensor, side_to_move: Tensor) -> Tensor:
        fixed = self.accumulators(indices)
        activated = torch.clamp(fixed, 0.0, 1.0).square()
        batch = torch.arange(indices.shape[0], device=indices.device)
        stm = activated[batch, side_to_move]
        non_stm = activated[batch, 1 - side_to_move]
        raw = self.output(torch.cat((stm, non_stm), dim=1)).squeeze(1)
        return raw * EVAL_SCALE_CP

    def deploy_parameter_count(self) -> int:
        return NUM_FEATURES * self.width + self.width + 2 * self.width + 1

    def quantised_size_bytes(self) -> int:
        return 2 * self.deploy_parameter_count()


def probability_loss(predicted_cp: Tensor, target_cp: Tensor) -> Tensor:
    """Bounded robust loss in win-probability space (no result blending)."""
    predicted_probability = torch.sigmoid(predicted_cp / EVAL_SCALE_CP)
    target_probability = torch.sigmoid(target_cp / EVAL_SCALE_CP)
    return torch.mean(torch.square(predicted_probability - target_probability))


class EncodedDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """In-memory reader for the compact participant-owned NPZ format."""

    def __init__(self, path: str | Path) -> None:
        loaded = np.load(path, allow_pickle=False)
        self.indices = loaded["indices"]
        self.sides = loaded["sides"]
        self.targets = loaded["targets"]
        if self.indices.dtype != np.uint16 or self.indices.shape[1:] != (2, 32):
            raise ValueError("indices must be uint16[N,2,32]")
        if np.any(self.indices > PADDING_FEATURE):
            raise ValueError("feature index exceeds padding sentinel")
        if len(self.indices) != len(self.sides) or len(self.indices) != len(self.targets):
            raise ValueError("dataset arrays have different lengths")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.from_numpy(self.indices[index].astype(np.int64, copy=False)),
            torch.tensor(int(self.sides[index]), dtype=torch.long),
            torch.tensor(float(self.targets[index]), dtype=torch.float32),
        )


@dataclass(frozen=True)
class CheckpointMetadata:
    width: int
    seed: int
    epoch: int
    validation_loss: float
    validation_mae_cp: float
    data_path: str


def save_checkpoint(
    path: str | Path, model: PerspectiveNNUE, metadata: CheckpointMetadata
) -> None:
    torch.save(
        {
            "format": "deepblue-nnue-float-v0",
            "state_dict": model.state_dict(),
            "metadata": metadata.__dict__,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[PerspectiveNNUE, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "deepblue-nnue-float-v0":
        raise ValueError("unsupported checkpoint format")
    metadata = dict(payload["metadata"])
    model = PerspectiveNNUE(int(metadata["width"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, metadata
