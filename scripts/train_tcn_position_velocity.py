from pathlib import Path
import sys
import random

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import create_dataloader
from src.models import TCNClassifier

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "val.csv"
)

LANDMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_position_velocity"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_SIZE = 300
NUM_CLASSES = 59

BATCH_SIZE = 8

HIDDEN_SIZE = 128

DROPOUT = 0.3

NUM_EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

LR_FACTOR = 0.5
LR_PATIENCE = 3

EARLY_STOP_PATIENCE = 7

GRAD_CLIP = 1.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MACRO F1
# ============================================================

def calculate_macro_f1(y_true, y_pred):

    f1_scores = []

    for class_id in range(NUM_CLASSES):

        true_positive = np.sum(
            (y_true == class_id)
            & (y_pred == class_id)
        )

        false_positive = np.sum(
            (y_true != class_id)
            & (y_pred == class_id)
        )

        false_negative = np.sum(
            (y_true == class_id)
            & (y_pred != class_id)
        )

        if (
            true_positive == 0
            and false_positive == 0
            and false_negative == 0
        ):
            f1 = 0.0

        else:

            precision_denominator = (
                true_positive + false_positive
            )

            recall_denominator = (
                true_positive + false_negative
            )

            precision = (
                true_positive
                / precision_denominator
                if precision_denominator > 0
                else 0.0
            )

            recall = (
                true_positive
                / recall_denominator
                if recall_denominator > 0
                else 0.0
            )

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = (
                    2
                    * precision
                    * recall
                    / (precision + recall)
                )

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(csv_path):

    df = pd.read_csv(csv_path)

    counts = (
        df["class_id"]
        .value_counts()
        .sort_index()
    )

    weights = np.ones(
        NUM_CLASSES,
        dtype=np.float32
    )

    total_samples = len(df)

    for class_id in range(NUM_CLASSES):

        class_count = counts.get(
            class_id,
            0
        )

        if class_count > 0:

            weights[class_id] = (
                total_samples
                / (NUM_CLASSES * class_count)
            )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for sequences, labels, lengths, padding_mask in loader:

        sequences = sequences.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        padding_mask = padding_mask.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            sequences,
            padding_mask
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * sequences.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    macro_f1 = calculate_macro_f1(
        np.array(all_labels),
        np.array(all_predictions)
    )

    accuracy = (
        np.mean(
            np.array(all_predictions)
            == np.array(all_labels)
        )
    )

    return (
        average_loss,
        accuracy,
        macro_f1
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for sequences, labels, lengths, padding_mask in loader:

            sequences = sequences.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            padding_mask = padding_mask.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                sequences,
                padding_mask
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * sequences.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )

    accuracy = np.mean(
        all_predictions == all_labels
    )

    macro_f1 = calculate_macro_f1(
        all_labels,
        all_predictions
    )

    return (
        average_loss,
        accuracy,
        macro_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TCN POSITION + VELOCITY TRAINING")
    print("=" * 70)

    print(f"Device           : {DEVICE}")
    print(f"Input size       : {INPUT_SIZE}")
    print(f"Classes          : {NUM_CLASSES}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Epochs           : {NUM_EPOCHS}")
    print(f"Learning rate    : {LEARNING_RATE}")
    print(f"Weight decay     : {WEIGHT_DECAY}")
    print()

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found:\n{TRAIN_CSV}"
        )

    if not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Validation CSV not found:\n{VAL_CSV}"
        )

    if not LANDMARK_DIR.exists():
        raise FileNotFoundError(
            f"Feature directory not found:\n{LANDMARK_DIR}"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_dataset, train_loader = create_dataloader(
        TRAIN_CSV,
        batch_size=BATCH_SIZE,
        shuffle=True,
        landmark_dir=LANDMARK_DIR,
    )

    val_dataset, val_loader = create_dataloader(
        VAL_CSV,
        batch_size=BATCH_SIZE,
        shuffle=False,
        landmark_dir=LANDMARK_DIR,
    )

    print(
        f"Training samples  : {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # Verify feature dimensions
    # --------------------------------------------------------

    sample_sequence, sample_label = (
        train_dataset[0]
    )

    print(
        f"Sample shape      : "
        f"{tuple(sample_sequence.shape)}"
    )

    if sample_sequence.shape[1] != INPUT_SIZE:

        raise ValueError(
            f"Expected {INPUT_SIZE} features/frame, "
            f"got {sample_sequence.shape[1]}"
        )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        TRAIN_CSV
    ).to(DEVICE)

    print(
        f"Class weights     : "
        f"min={class_weights.min().item():.4f}, "
        f"max={class_weights.max().item():.4f}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TCNClassifier(
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters        : {parameter_count:,}"
    )

    # --------------------------------------------------------
    # Loss / optimizer / scheduler
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_f1 = -1.0
    best_epoch = 0

    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "tcn_position_velocity_best.pt"
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        train_loss, train_acc, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        val_loss, val_acc, val_f1 = (
            validate(
                model,
                val_loader,
                criterion
            )
        )

        scheduler.step(
            val_f1
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"LR {current_lr:.6f} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Train F1 {train_f1:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "config": {
                    "input_size": INPUT_SIZE,
                    "num_classes": NUM_CLASSES,
                    "batch_size": BATCH_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "dropout": DROPOUT,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                    "feature_representation": (
                        "position_plus_velocity"
                    ),
                },
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print(
                f"  -> New best validation F1: "
                f"{best_val_f1:.4f}"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Macro F1 : "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Best epoch               : "
        f"{best_epoch}"
    )

    print(
        f"Checkpoint               : "
        f"{checkpoint_path}"
    )

    print()
    print(
        "Baseline TCN remains unchanged."
    )


if __name__ == "__main__":
    main()