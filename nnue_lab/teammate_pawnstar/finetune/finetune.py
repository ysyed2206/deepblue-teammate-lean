"""Conservative fine-tune of the Pawnstar-v12-initialized model on our own
engine-annotated (Stockfish via Lichess CC0) data.

Per the mission's own fine-tuning guardrails (``TEAMMATE_NNUE_MISSION.md``
section 14): preserve the original donor checkpoint (never overwritten --
this script only ever reads ``checkpoints/pawnstar_v12_dequantized_float.pt``
and writes to a different path), low learning rate, frequent validation,
checkpoint often, avoid catastrophic forgetting.

Reuses ``nnue_lab.model``'s ``PerspectiveNNUE``/``probability_loss`` and
``nnue_lab.train``'s ``validate``/dataset loader unmodified -- only the
*initialization* (from the converted donor checkpoint, not random) and the
hyperparameters (much lower LR, since this starts from an already-good
network rather than from scratch) differ from a from-scratch run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

from nnue_lab.model import CheckpointMetadata, PerspectiveNNUE, probability_loss, save_checkpoint  # noqa: E402
from nnue_lab.train import load_dataset, validate  # noqa: E402

DONOR_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "pawnstar_v12_dequantized_float.pt"
OUT_DIR = Path(__file__).resolve().parent / "checkpoints"


def load_donor_initialized_model() -> PerspectiveNNUE:
    payload = torch.load(DONOR_CHECKPOINT, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    model = PerspectiveNNUE(int(metadata["width"]))
    model.load_state_dict(payload["state_dict"])
    return model


def main() -> None:
    train_data = Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\finetune_data\train.npz")
    validation_data = Path(r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\finetune_data\validation.npz")

    seed = 20260904
    epochs = 4
    batch_size = 512
    validation_batch_size = 1024
    # Much lower than a from-scratch run's 0.002 -- this network already
    # encodes real evaluation knowledge from Pawnstar's own large-scale
    # training; a low LR nudges it toward our data without catastrophic
    # forgetting.
    lr = 0.0002
    threads = 6
    patience = 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)

    train_dataset = load_dataset(train_data)
    validation_dataset = load_dataset(validation_data)
    generator = torch.Generator().manual_seed(seed)
    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator
    )
    validation_loader: DataLoader = DataLoader(
        validation_dataset, batch_size=validation_batch_size, shuffle=False, num_workers=0
    )

    model = load_donor_initialized_model()
    print(f"loaded donor-initialized model, width={model.width}")

    # Pre-fine-tune baseline: how good is the donor network on OUR data,
    # before any training? This is the number fine-tuning must beat.
    baseline_loss, baseline_mae = validate(model, validation_loader)
    print(f"pre-finetune baseline: validation_probability_mse={baseline_loss:.6f} validation_teacher_mae_cp={baseline_mae:.3f}")

    sparse_optimizer = torch.optim.SparseAdam([model.feature_weights.weight], lr=lr)
    dense_optimizer = torch.optim.AdamW([model.feature_bias, *model.output.parameters()], lr=lr, weight_decay=1e-5)

    best_loss = baseline_loss
    best_epoch = 0
    epochs_without_improvement = 0
    history = [{"event": "pre_finetune_baseline", "validation_probability_mse": baseline_loss, "validation_teacher_mae_cp": baseline_mae}]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT_DIR / "finetuned_best.pt"
    log_path = OUT_DIR / "finetune_train.jsonl"
    run_started = time.perf_counter()

    # Preserve the pre-finetune baseline as its own checkpoint in case fine-tuning
    # never improves on it -- "never overwrite the original donor checkpoint" also
    # means always having a fallback at least as good as the untouched conversion.
    save_checkpoint(
        checkpoint_path,
        model,
        CheckpointMetadata(width=model.width, seed=seed, epoch=0, validation_loss=baseline_loss, validation_mae_cp=baseline_mae, data_path=str(train_data)),
    )

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(json.dumps(history[0]) + "\n")
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_started = time.perf_counter()
            loss_sum = 0.0
            seen = 0
            for indices, sides, targets in train_loader:
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
            validation_loss, validation_mae = validate(model, validation_loader)
            improved = validation_loss < best_loss - 1e-7
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_path,
                    model,
                    CheckpointMetadata(width=model.width, seed=seed, epoch=epoch, validation_loss=validation_loss, validation_mae_cp=validation_mae, data_path=str(train_data)),
                )
            else:
                epochs_without_improvement += 1
            record = {
                "event": "epoch",
                "epoch": epoch,
                "train_probability_mse": loss_sum / seen,
                "validation_probability_mse": validation_loss,
                "validation_teacher_mae_cp": validation_mae,
                "training_seconds": training_seconds,
                "examples_per_second": seen / training_seconds,
                "best": improved,
            }
            history.append(record)
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(json.dumps(record))
            if epochs_without_improvement >= patience:
                print("early stop: no improvement")
                break

    summary = {
        "format": "deepblue-nnue-finetune-summary-v0",
        "donor_checkpoint": str(DONOR_CHECKPOINT),
        "baseline_validation_probability_mse": baseline_loss,
        "baseline_validation_teacher_mae_cp": baseline_mae,
        "best_epoch": best_epoch,
        "best_validation_probability_mse": best_loss,
        "improved_over_baseline": best_loss < baseline_loss,
        "checkpoint": str(checkpoint_path),
        "total_wall_seconds": time.perf_counter() - run_started,
    }
    (OUT_DIR / "finetune_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
