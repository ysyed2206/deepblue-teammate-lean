"""Trainer for the independent sparse SCReLU v0 network. CPU by default;
pass --device cuda to train on a GPU. Checkpoints are always loaded back
with map_location, so a GPU-trained checkpoint drops into the CPU pipeline
with no extra step."""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnue_lab.model import (
    CheckpointMetadata,
    PerspectiveNNUE,
    probability_loss,
    save_checkpoint,
)


def load_dataset(path: Path) -> TensorDataset:
    loaded = np.load(path, allow_pickle=False)
    indices = torch.from_numpy(loaded["indices"].astype(np.int64, copy=True))
    sides = torch.from_numpy(loaded["sides"].astype(np.int64, copy=True))
    targets = torch.from_numpy(loaded["targets"].astype(np.float32, copy=True))
    if indices.ndim != 3 or indices.shape[1:] != (2, 32):
        raise ValueError(f"bad feature shape in {path}: {indices.shape}")
    return TensorDataset(indices, sides, targets)


@torch.no_grad()
def validate(model: PerspectiveNNUE, loader: DataLoader[Any], device: str) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    absolute_error_sum = 0.0
    count = 0
    for indices, sides, targets in loader:
        indices, sides, targets = indices.to(device), sides.to(device), targets.to(device)
        predictions = model(indices, sides)
        batch = int(targets.shape[0])
        loss_sum += float(probability_loss(predictions, targets)) * batch
        absolute_error_sum += float(torch.abs(predictions - targets).sum())
        count += batch
    return loss_sum / count, absolute_error_sum / count


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Either the machine has no NVIDIA GPU, no driver, or torch was installed "
            "without CUDA support (reinstall from https://pytorch.org/get-started/locally/ "
            "picking a CUDA version). Falling back silently would just train on the CPU "
            "under a false name, so this stops instead."
        )
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(4, args.threads)))

    train_dataset = load_dataset(args.train_data)
    validation_dataset = load_dataset(args.validation_data)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
    validation_loader: DataLoader[Any] = DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = PerspectiveNNUE(args.width).to(args.device)
    sparse_optimizer = torch.optim.SparseAdam([model.feature_weights.weight], lr=args.lr)
    dense_optimizer = torch.optim.AdamW(
        [model.feature_bias, *model.output.parameters()],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    total_examples = 0
    total_training_seconds = 0.0
    run_started = time.perf_counter()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8", newline="\n") as log_handle:
        header = {
            "event": "start",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": " ".join(shlex.quote(part) for part in sys.argv),
            "seed": args.seed,
            "width": args.width,
            "train_data": str(args.train_data.resolve()),
            "validation_data": str(args.validation_data.resolve()),
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "batch_size": args.batch_size,
            "threads": args.threads,
            "torch_version": torch.__version__,
        }
        log_handle.write(json.dumps(header) + "\n")
        log_handle.flush()
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_started = time.perf_counter()
            loss_sum = 0.0
            seen = 0
            for indices, sides, targets in train_loader:
                indices, sides, targets = (
                    indices.to(args.device), sides.to(args.device), targets.to(args.device),
                )
                sparse_optimizer.zero_grad(set_to_none=True)
                dense_optimizer.zero_grad(set_to_none=True)
                predictions = model(indices, sides)
                loss = probability_loss(predictions, targets)
                loss.backward()
                sparse_optimizer.step()
                dense_optimizer.step()
                batch = int(targets.shape[0])
                loss_sum += float(loss.detach()) * batch
                seen += batch
            training_seconds = time.perf_counter() - epoch_started
            total_training_seconds += training_seconds
            total_examples += seen
            validation_started = time.perf_counter()
            validation_loss, validation_mae = validate(model, validation_loader, args.device)
            validation_seconds = time.perf_counter() - validation_started
            improved = validation_loss < best_loss - args.min_delta
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                metadata = CheckpointMetadata(
                    width=args.width,
                    seed=args.seed,
                    epoch=epoch,
                    validation_loss=validation_loss,
                    validation_mae_cp=validation_mae,
                    data_path=str(args.train_data.resolve()),
                )
                save_checkpoint(args.checkpoint, model, metadata)
            else:
                epochs_without_improvement += 1
            record = {
                "event": "epoch",
                "epoch": epoch,
                "train_probability_mse": loss_sum / seen,
                "validation_probability_mse": validation_loss,
                "validation_teacher_mae_cp": validation_mae,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "examples_per_second": seen / training_seconds,
                "best": improved,
            }
            history.append(record)
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(json.dumps(record))
            if epochs_without_improvement >= args.patience:
                break

    summary: dict[str, Any] = {
        "format": "deepblue-nnue-training-summary-v0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "seed": args.seed,
        "width": args.width,
        "parameter_count": model.deploy_parameter_count(),
        "float_parameter_bytes": model.deploy_parameter_count() * 4,
        "expected_quantised_bytes": model.quantised_size_bytes(),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_probability_mse": best_loss,
        "best_validation_teacher_mae_cp": next(
            record["validation_teacher_mae_cp"]
            for record in history
            if record["epoch"] == best_epoch
        ),
        "training_seconds": total_training_seconds,
        "total_wall_seconds": time.perf_counter() - run_started,
        "mean_training_examples_per_second": total_examples / total_training_seconds,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "log": str(args.log.resolve()),
        "history": history,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--device", type=str, default="cpu",
                         help="'cpu' or 'cuda' (needs an NVIDIA GPU + CUDA-enabled torch)")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    lab = Path(__file__).resolve().parent
    for output in (args.checkpoint, args.log, args.summary):
        if not output.resolve().is_relative_to(lab):
            parser.error("model, log, and summary outputs must stay inside nnue_lab")
        output.parent.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    train(parse_args())

