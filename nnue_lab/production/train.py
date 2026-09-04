"""Deterministic CPU trainer for the production Perspective Chess768 NNUE."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nnue_lab.production.artifacts import (
    PRODUCTION_FLOAT_FORMAT,
    atomic_torch_save,
    load_raw_checkpoint,
)
from nnue_lab.production.dataset import (
    fingerprint_dataset,
    load_encoded_npz,
    reject_pristine_before_selection,
)
from nnue_lab.production.features import MAX_PIECES, NUM_FEATURES, NUM_PERSPECTIVES
from nnue_lab.production.model import (
    EVALUATION_SCALE_CP,
    PerspectiveChess768NNUE,
    probability_mse,
)

LAB_DIR = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def command_line() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def parse_milestones(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    milestones = tuple(sorted({int(part) for part in value.split(",")}))
    if any(epoch <= 0 for epoch in milestones):
        raise ValueError("learning-rate milestones must be positive epochs")
    return milestones


def configure_determinism(seed: int, threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(max(1, min(4, threads)))
    torch.use_deterministic_algorithms(True)


def dataset_manifest(
    paths: list[Path], legacy_mod768: bool
) -> list[dict[str, Any]]:
    return [
        fingerprint_dataset(
            path, legacy_chessbuckets_mod768=legacy_mod768
        ).__dict__
        for path in paths
    ]


def _limited_count(fingerprints: list[dict[str, Any]], limit: int | None) -> int:
    count = sum(int(item["examples"]) for item in fingerprints)
    return count if limit is None else min(count, limit)


def iter_training_batches(
    paths: list[Path],
    *,
    batch_size: int,
    rng: np.random.Generator,
    legacy_mod768: bool,
    maximum_examples: int | None,
):
    """Yield deterministic shuffled batches while holding only one shard."""
    path_order = rng.permutation(len(paths))
    remaining = maximum_examples
    for path_index in path_order:
        arrays = load_encoded_npz(
            paths[int(path_index)], legacy_chessbuckets_mod768=legacy_mod768
        )
        order = rng.permutation(arrays.count)
        if remaining is not None:
            order = order[:remaining]
        for start in range(0, len(order), batch_size):
            selection = order[start : start + batch_size]
            indices = torch.from_numpy(
                arrays.indices[selection].astype(np.int64, copy=False)
            )
            sides = torch.from_numpy(
                arrays.sides[selection].astype(np.int64, copy=False)
            )
            targets = torch.from_numpy(
                arrays.targets[selection].astype(np.float32, copy=False)
            )
            yield indices, sides, targets
        if remaining is not None:
            remaining -= len(order)
            if remaining <= 0:
                return


@torch.no_grad()
def validate(
    model: PerspectiveChess768NNUE,
    paths: list[Path],
    *,
    batch_size: int,
    legacy_mod768: bool,
    maximum_examples: int | None,
) -> tuple[float, float, int, float]:
    model.eval()
    loss_sum = 0.0
    absolute_error_sum = 0.0
    seen = 0
    started = time.perf_counter()
    remaining = maximum_examples
    for path in paths:
        limit = remaining
        arrays = load_encoded_npz(
            path,
            legacy_chessbuckets_mod768=legacy_mod768,
            limit=limit,
        )
        for start in range(0, arrays.count, batch_size):
            end = min(start + batch_size, arrays.count)
            indices = torch.from_numpy(
                arrays.indices[start:end].astype(np.int64, copy=False)
            )
            sides = torch.from_numpy(
                arrays.sides[start:end].astype(np.int64, copy=False)
            )
            targets = torch.from_numpy(arrays.targets[start:end])
            predictions = model(indices, sides)
            batch = end - start
            loss_sum += float(probability_mse(predictions, targets)) * batch
            absolute_error_sum += float(torch.abs(predictions - targets).sum())
            seen += batch
        if remaining is not None:
            remaining -= arrays.count
            if remaining <= 0:
                break
    elapsed = time.perf_counter() - started
    if seen == 0:
        raise ValueError("validation set is empty")
    return loss_sum / seen, absolute_error_sum / seen, seen, elapsed


def architecture_metadata(width: int) -> dict[str, Any]:
    return {
        "name": "PerspectiveChess768-SCReLU",
        "width": width,
        "num_features": NUM_FEATURES,
        "perspectives": NUM_PERSPECTIVES,
        "max_active_features_per_perspective": MAX_PIECES,
        "shared_feature_transformer": True,
        "accumulator_order": ["white", "black"],
        "tail_order": ["side_to_move", "non_side_to_move"],
        "activation": "clamp(x,0,1)^2",
        "output": "single scalar centipawns from side-to-move perspective",
        "evaluation_scale_cp": EVALUATION_SCALE_CP,
        "parameter_count": NUM_FEATURES * width + width + 2 * width + 1,
    }


def checkpoint_payload(
    model: PerspectiveChess768NNUE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
    *,
    config: dict[str, Any],
    data: dict[str, Any],
    training_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": PRODUCTION_FLOAT_FORMAT,
        "version": 1,
        "created_utc": utc_now(),
        "architecture": architecture_metadata(model.width),
        "objective": {
            "teacher": "Stockfish static cp supplied by dataset",
            "target_orientation": "side-to-move relative",
            "loss": "MSE(sigmoid(predicted_cp/400), sigmoid(target_cp/400))",
            "game_result_blending": False,
        },
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "data": data,
        "training_state": training_state,
        "rng_state": {
            "python": random.getstate(),
            "numpy_legacy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "data_order": training_state.get("data_order_rng_state"),
            "data_order_derivation": "PCG64(seed + 1000003 * global_epoch)",
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def _resume_compatible(saved: dict[str, Any], current: dict[str, Any]) -> None:
    immutable = (
        "width",
        "seed",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "lr_milestones",
        "lr_gamma",
        "legacy_chessbuckets_mod768",
        "maximum_train_examples",
        "maximum_validation_examples",
    )
    mismatches = [key for key in immutable if saved.get(key) != current.get(key)]
    if mismatches:
        raise ValueError(f"resume configuration differs for {mismatches}")


def _validate_data_expansion(
    saved: dict[str, Any], current: dict[str, Any], allow_expansion: bool
) -> None:
    if saved == current:
        return
    if not allow_expansion:
        raise ValueError(
            "resume data fingerprints differ; use --allow-data-expansion only when "
            "adding immutable training shards as a new stage"
        )
    if saved.get("validation") != current.get("validation"):
        raise ValueError("validation fingerprints cannot change across data stages")
    old_train = {
        (item["path"], item["sha256"]): item for item in saved.get("train", [])
    }
    new_train = {
        (item["path"], item["sha256"]): item for item in current.get("train", [])
    }
    if not old_train.keys() <= new_train.keys():
        raise ValueError("data expansion removed or changed a prior training shard")
    if len(new_train) <= len(old_train):
        raise ValueError("--allow-data-expansion requires at least one new training shard")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism(args.seed, args.threads)
    train_paths = [path.resolve() for path in args.train_data]
    validation_paths = [path.resolve() for path in args.validation_data]
    reject_pristine_before_selection(train_paths + validation_paths, False)
    train_fingerprints = dataset_manifest(train_paths, args.legacy_chessbuckets_mod768)
    validation_fingerprints = dataset_manifest(
        validation_paths, args.legacy_chessbuckets_mod768
    )
    train_count = _limited_count(train_fingerprints, args.maximum_train_examples)
    validation_count = _limited_count(
        validation_fingerprints, args.maximum_validation_examples
    )
    if train_count == 0 or validation_count == 0:
        raise ValueError("training and validation data must both be non-empty")

    milestones = parse_milestones(args.lr_milestones)
    config: dict[str, Any] = {
        "command": command_line(),
        "stage_name": args.stage_name,
        "width": args.width,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_milestones": list(milestones),
        "lr_gamma": args.lr_gamma,
        "patience": args.patience,
        "minimum_delta": args.minimum_delta,
        "threads": args.threads,
        "legacy_chessbuckets_mod768": args.legacy_chessbuckets_mod768,
        "maximum_train_examples": args.maximum_train_examples,
        "maximum_validation_examples": args.maximum_validation_examples,
    }
    data = {
        "train": train_fingerprints,
        "validation": validation_fingerprints,
        "train_examples_per_epoch": train_count,
        "validation_examples": validation_count,
    }

    model = PerspectiveChess768NNUE(args.width)
    # The model defaults to sparse gradients for runtime-prototype flexibility.
    # Chess768's table is small and virtually every row occurs in a large batch;
    # dense gradients let the requested AdamW update all parameters consistently.
    model.feature_weights.sparse = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(milestones), gamma=args.lr_gamma
    )
    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    best_mae = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    stage_history: list[dict[str, Any]] = []
    total_train_seconds = 0.0
    total_examples_seen = 0

    latest_path = args.run_dir / "latest.pt"
    best_path = args.run_dir / "best.pt"
    log_path = args.run_dir / "train.jsonl"
    summary_path = args.run_dir / "training_summary.json"
    config_path = args.run_dir / "config.json"
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"resume requested but {latest_path} does not exist")
        payload = load_raw_checkpoint(latest_path)
        if payload.get("format") != PRODUCTION_FLOAT_FORMAT:
            raise ValueError("resume checkpoint has the wrong format")
        _resume_compatible(dict(payload["config"]), config)
        saved_data = dict(payload["data"])
        data_expanded = saved_data != data
        _validate_data_expansion(saved_data, data, args.allow_data_expansion)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        state = dict(payload["training_state"])
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state["best_validation_probability_mse"])
        best_epoch = int(state["best_epoch"])
        best_mae = float(state["best_validation_teacher_mae_cp"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        if data_expanded:
            epochs_without_improvement = 0
        history = list(state.get("history", []))
        stage_history = list(state.get("stage_history", []))
        total_train_seconds = float(state.get("total_training_seconds", 0.0))
        total_examples_seen = int(state.get("total_examples_seen", 0))
        rng_state = dict(payload.get("rng_state", {}))
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy_legacy"])
            torch.set_rng_state(rng_state["torch_cpu"])
    elif args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(
            f"run directory is not empty: {args.run_dir}; use --resume only for its latest.pt"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stage_history.append(
        {
            "stage_name": args.stage_name,
            "started_utc": utc_now(),
            "start_epoch": start_epoch,
            "train_examples_per_epoch": train_count,
            "train_shards": train_fingerprints,
            "allow_data_expansion": args.allow_data_expansion,
            "stage_index": len(stage_history) + 1,
            "command": command_line(),
        }
    )
    config_path.write_text(
        json.dumps({"config": config, "data": data}, indent=2) + "\n",
        encoding="utf-8",
    )

    run_started = time.perf_counter()
    log_mode = "a" if args.resume else "w"
    with log_path.open(log_mode, encoding="utf-8", newline="\n") as log_handle:
        start_record = {
            "event": "resume" if args.resume else "start",
            "created_utc": utc_now(),
            "start_epoch": start_epoch,
            "config": config,
            "data": data,
        }
        log_handle.write(json.dumps(start_record) + "\n")
        log_handle.flush()
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            train_started = time.perf_counter()
            loss_sum = 0.0
            seen = 0
            learning_rate = float(optimizer.param_groups[0]["lr"])
            data_order_rng = np.random.default_rng(args.seed + 1_000_003 * epoch)
            for indices, sides, targets in iter_training_batches(
                train_paths,
                batch_size=args.batch_size,
                rng=data_order_rng,
                legacy_mod768=args.legacy_chessbuckets_mod768,
                maximum_examples=args.maximum_train_examples,
            ):
                optimizer.zero_grad(set_to_none=True)
                predictions = model(indices, sides)
                loss = probability_mse(predictions, targets)
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch}, examples {seen}; "
                        "the previous complete checkpoint is preserved"
                    )
                loss.backward()
                optimizer.step()
                batch = int(targets.shape[0])
                loss_sum += loss_value * batch
                seen += batch
            training_seconds = time.perf_counter() - train_started
            validation_loss, validation_mae, validation_seen, validation_seconds = (
                validate(
                    model,
                    validation_paths,
                    batch_size=args.validation_batch_size,
                    legacy_mod768=args.legacy_chessbuckets_mod768,
                    maximum_examples=args.maximum_validation_examples,
                )
            )
            if not math.isfinite(validation_loss) or not math.isfinite(validation_mae):
                raise FloatingPointError(
                    f"non-finite validation metrics at epoch {epoch}; "
                    "the previous complete checkpoint is preserved"
                )
            scheduler.step()
            improved = validation_loss < best_loss - args.minimum_delta
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
                best_mae = validation_mae
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            total_train_seconds += training_seconds
            total_examples_seen += seen
            record = {
                "event": "epoch",
                "epoch": epoch,
                "learning_rate": learning_rate,
                "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_probability_mse": loss_sum / seen,
                "validation_probability_mse": validation_loss,
                "validation_teacher_mae_cp": validation_mae,
                "training_examples": seen,
                "validation_examples": validation_seen,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "training_examples_per_second": seen / training_seconds,
                "validation_examples_per_second": validation_seen
                / validation_seconds,
                "best": improved,
            }
            history.append(record)
            training_state = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_probability_mse": best_loss,
                "best_validation_teacher_mae_cp": best_mae,
                "epochs_without_improvement": epochs_without_improvement,
                "total_training_seconds": total_train_seconds,
                "total_examples_seen": total_examples_seen,
                "history": history,
                "stage_history": stage_history,
                "data_order_rng_state": data_order_rng.bit_generator.state,
            }
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                config=config,
                data=data,
                training_state=training_state,
            )
            atomic_torch_save(payload, latest_path)
            if improved:
                atomic_torch_save(payload, best_path)
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(json.dumps(record), flush=True)
            if epochs_without_improvement >= args.patience:
                break

    if not history or not best_path.exists():
        raise RuntimeError("training produced no valid best checkpoint")
    summary: dict[str, Any] = {
        "format": "deepblue-perspective-chess768-training-summary-v1",
        "created_utc": utc_now(),
        "architecture": architecture_metadata(args.width),
        "objective": (
            "MSE(sigmoid(predicted_cp/400), sigmoid(side_to_move_teacher_cp/400)); "
            "no game-result blending"
        ),
        "config": config,
        "data": data,
        "stage_history": stage_history,
        "epochs_completed_this_invocation": max(0, len(history) - (start_epoch - 1)),
        "epochs_completed_total": len(history),
        "best_epoch": best_epoch,
        "best_validation_probability_mse": best_loss,
        "best_validation_teacher_mae_cp": best_mae,
        "total_training_seconds": total_train_seconds,
        "invocation_wall_seconds": time.perf_counter() - run_started,
        "mean_training_examples_per_second": total_examples_seen
        / total_train_seconds,
        "total_examples_seen": total_examples_seen,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_bytes": best_path.stat().st_size,
        "latest_checkpoint": str(latest_path.resolve()),
        "latest_checkpoint_bytes": latest_path.stat().st_size,
        "log": str(log_path.resolve()),
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-data", type=Path, nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(256, 512), required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--validation-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-milestones", default="")
    parser.add_argument("--lr-gamma", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--stage-name", default="stage1")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--minimum-delta", type=float, default=1e-7)
    parser.add_argument("--maximum-train-examples", type=int)
    parser.add_argument("--maximum-validation-examples", type=int)
    parser.add_argument(
        "--legacy-chessbuckets-mod768",
        action="store_true",
        help=(
            "explicit smoke bridge for old 6144-row datasets; never use for native "
            "production data"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-data-expansion",
        action="store_true",
        help="on resume, allow only append-only training-shard expansion; validation stays frozen",
    )
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_relative_to(LAB_DIR):
        parser.error("run directory must stay inside nnue_lab")
    positive = (
        args.epochs,
        args.batch_size,
        args.validation_batch_size,
        args.learning_rate,
        args.threads,
        args.patience,
    )
    if any(value <= 0 for value in positive):
        parser.error("epochs, batches, learning rate, threads, and patience must be positive")
    for value in (args.maximum_train_examples, args.maximum_validation_examples):
        if value is not None and value <= 0:
            parser.error("example limits must be positive")
    try:
        parse_milestones(args.lr_milestones)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    run_training(parse_args())
