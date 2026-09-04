"""Self-describing float-checkpoint helpers for production and baseline nets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nnue_lab.production.model import PerspectiveChess768NNUE

PRODUCTION_FLOAT_FORMAT = "deepblue-perspective-chess768-float-v1"
BASELINE_FLOAT_FORMAT = "deepblue-nnue-float-v0"


@dataclass(frozen=True)
class LoadedFloatModel:
    model: nn.Module
    format_name: str
    width: int
    num_features: int
    padding_feature: int
    metadata: dict[str, Any]


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_raw_checkpoint(path: Path, device: str = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: checkpoint payload is not a dictionary")
    return payload


def load_float_model(path: Path, device: str = "cpu") -> LoadedFloatModel:
    """Load either the production Chess768 model or preserved H128 baseline."""
    payload = load_raw_checkpoint(path, device)
    format_name = str(payload.get("format", ""))
    if format_name == PRODUCTION_FLOAT_FORMAT:
        architecture = dict(payload.get("architecture", {}))
        width = int(architecture.get("width", 0))
        if int(architecture.get("num_features", 0)) != 768:
            raise ValueError(f"{path}: production feature vocabulary is not 768")
        model: nn.Module = PerspectiveChess768NNUE(width)
        model.load_state_dict(payload["state_dict"])
        metadata = {
            "architecture": architecture,
            "training_state": dict(payload.get("training_state", {})),
            "config": dict(payload.get("config", {})),
            "data": dict(payload.get("data", {})),
        }
        num_features = 768
        padding_feature = 768
    elif format_name == BASELINE_FLOAT_FORMAT:
        from nnue_lab.model import PerspectiveNNUE

        metadata = dict(payload.get("metadata", {}))
        width = int(metadata.get("width", 0))
        model = PerspectiveNNUE(width)
        model.load_state_dict(payload["state_dict"])
        num_features = 6144
        padding_feature = 6144
    else:
        raise ValueError(f"{path}: unsupported float checkpoint format {format_name!r}")
    model.to(device)
    model.eval()
    return LoadedFloatModel(
        model=model,
        format_name=format_name,
        width=width,
        num_features=num_features,
        padding_feature=padding_feature,
        metadata=metadata,
    )
