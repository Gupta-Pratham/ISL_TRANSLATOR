import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.dataset import create_dataloader
from src.models import LSTMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = r".\data\processed\train.csv"
VAL_CSV = r".\data\processed\val.csv"

CHECKPOINT_DIR = Path(
    r".\models\checkpoints"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "lstm_baseline_best.pt"
)

SEED = 42

BATCH_SIZE = 8

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 7


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(csv_file, num_classes):

    df = pd.read_csv(csv_file)

    counts = (
        df["class_id"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    weights = []

    for class_id in range(num_classes):

        count = counts.get(
            class_id,
            0
        )

        if count == 0:
            weight = 0.0
        else:
            weight = total / (
                num_classes * count
            )

        weights.append(weight)

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
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for (
        sequences,
        labels,
        lengths,
        padding_mask
    ) in loader:

        sequences = sequences.to(device)

        labels = labels.to(device)

        padding_mask = padding_mask.to(device)

        optimizer.zero_grad()

        logits = model(
            sequences,
            padding_mask
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * sequences.size(0)
        )

        predictions = (
            logits.argmax(dim=1)
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

    avg_loss = (
        total_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return avg_loss, accuracy, macro_f1


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for (
            sequences,
            labels,
            lengths,
            padding_mask
        ) in loader:

            sequences = sequences.to(device)

            labels = labels.to(device)

            padding_mask = padding_mask.to(device)

            logits = model(
                sequences,
                padding_mask
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * sequences.size(0)
            )

            predictions = (
                logits.argmax(dim=1)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = (
        total_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return avg_loss, accuracy, macro_f1


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("============================================")
    print("LSTM Baseline Training")
    print("============================================")

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    _, train_loader = create_dataloader(
        TRAIN_CSV,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    _, val_loader = create_dataloader(
        VAL_CSV,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LSTMClassifier(
        input_size=150,
        hidden_size=128,
        num_layers=2,
        num_classes=8,
        dropout=0.3
    ).to(device)

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        TRAIN_CSV,
        num_classes=8
    ).to(device)

    print(
        "Class weights:",
        class_weights.detach().cpu().numpy()
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_f1 = -1.0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_f1
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"LR={current_lr:.6f} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train Acc={train_acc:.4f} | "
            f"Train F1={train_f1:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Acc={val_acc:.4f} | "
            f"Val F1={val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_val_f1,
                    "epoch": epoch,
                },
                CHECKPOINT_PATH
            )

            print(
                f"  → Saved best model "
                f"(Val F1={best_val_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    print("\n============================================")
    print("Training complete")
    print("============================================")

    print(
        f"Best validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )


if __name__ == "__main__":
    main()