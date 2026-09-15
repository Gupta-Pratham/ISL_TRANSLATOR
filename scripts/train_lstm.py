import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.dataset import create_dataloader
from src.models import LSTMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 8
EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 7
GRAD_CLIP = 1.0


# ------------------------------------------------------------
# LSTM architecture
# ------------------------------------------------------------

INPUT_SIZE = 150
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 59
DROPOUT = 0.3


# ============================================================
# DATA PATHS
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
    / "landmarks_preprocessed"
)


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "lstm_baseline_best.pt"
)


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
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    csv_file,
    num_classes
):

    df = pd.read_csv(
        csv_file
    )

    counts = (
        df["class_id"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    weights = []

    for class_id in range(
        num_classes
    ):

        count = counts.get(
            class_id,
            0
        )

        if count == 0:

            weight = 0.0

        else:

            weight = (
                total
                / (
                    num_classes
                    * count
                )
            )

        weights.append(
            weight
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

        sequences = sequences.to(
            device
        )

        labels = labels.to(
            device
        )

        padding_mask = padding_mask.to(
            device
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            sequences,
            padding_mask
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP
        )


        optimizer.step()


        # ----------------------------------------------------
        # Accumulate loss
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            * sequences.size(0)
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = (
            logits.argmax(
                dim=1
            )
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


    # ========================================================
    # METRICS
    # ========================================================

    avg_loss = (
        total_loss
        / len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    _, _, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(
                range(NUM_CLASSES)
            ),
            average="macro",
            zero_division=0
        )
    )


    return (
        avg_loss,
        accuracy,
        macro_f1
    )


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

            sequences = sequences.to(
                device
            )

            labels = labels.to(
                device
            )

            padding_mask = padding_mask.to(
                device
            )


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(
                sequences,
                padding_mask
            )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels
            )


            total_loss += (
                loss.item()
                * sequences.size(0)
            )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = (
                logits.argmax(
                    dim=1
                )
            )


            all_predictions.extend(
                predictions.cpu()
                .numpy()
            )

            all_labels.extend(
                labels.cpu()
                .numpy()
            )


    # ========================================================
    # METRICS
    # ========================================================

    avg_loss = (
        total_loss
        / len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    _, _, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(
                range(NUM_CLASSES)
            ),
            average="macro",
            zero_division=0
        )
    )


    return (
        avg_loss,
        accuracy,
        macro_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        SEED
    )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print(
        "============================================"
    )

    print(
        "LSTM BASELINE TRAINING"
    )

    print(
        "============================================"
    )

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Input features: {INPUT_SIZE}"
    )

    print(
        f"Hidden size: {HIDDEN_SIZE}"
    )

    print(
        f"LSTM layers: {NUM_LAYERS}"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    print(
        f"Dropout: {DROPOUT}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        f"Gradient clip: {GRAD_CLIP}"
    )

    print(
        "============================================"
    )


    # ========================================================
    # CHECK PATHS
    # ========================================================

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"Training CSV not found:\n"
            f"{TRAIN_CSV}"
        )

    if not VAL_CSV.exists():

        raise FileNotFoundError(
            f"Validation CSV not found:\n"
            f"{VAL_CSV}"
        )

    if not LANDMARK_DIR.exists():

        raise FileNotFoundError(
            f"Landmark directory not found:\n"
            f"{LANDMARK_DIR}"
        )


    # ========================================================
    # LOAD DATA
    # ========================================================

    train_dataset, train_loader = (
        create_dataloader(
            TRAIN_CSV,
            batch_size=BATCH_SIZE,
            shuffle=True,
            landmark_dir=LANDMARK_DIR
        )
    )


    val_dataset, val_loader = (
        create_dataloader(
            VAL_CSV,
            batch_size=BATCH_SIZE,
            shuffle=False,
            landmark_dir=LANDMARK_DIR
        )
    )


    print()
    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )


    # ========================================================
    # VALIDATE LABELS
    # ========================================================

    train_labels = np.array([
        train_dataset[i][1]
        for i in range(
            len(train_dataset)
        )
    ])

    val_labels = np.array([
        val_dataset[i][1]
        for i in range(
            len(val_dataset)
        )
    ])


    if len(train_labels) == 0:

        raise RuntimeError(
            "Training dataset is empty."
        )

    if len(val_labels) == 0:

        raise RuntimeError(
            "Validation dataset is empty."
        )


    if np.any(
        train_labels < 0
    ) or np.any(
        train_labels >= NUM_CLASSES
    ):

        raise ValueError(
            "Training dataset contains "
            "invalid class IDs."
        )


    if np.any(
        val_labels < 0
    ) or np.any(
        val_labels >= NUM_CLASSES
    ):

        raise ValueError(
            "Validation dataset contains "
            "invalid class IDs."
        )


    # ========================================================
    # MODEL
    # ========================================================

    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)


    # ========================================================
    # MODEL PARAMETERS
    # ========================================================

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )


    print()
    print(
        f"Model parameters: "
        f"{num_params:,}"
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            TRAIN_CSV,
            num_classes=NUM_CLASSES
        )
        .to(DEVICE)
    )


    print()
    print(
        "Class counts:"
    )

    class_counts = np.bincount(
        train_labels,
        minlength=NUM_CLASSES
    )

    print(
        class_counts.tolist()
    )

    print()
    print(
        "Class weights:"
    )

    print(
        class_weights
        .detach()
        .cpu()
        .numpy()
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )


    # ========================================================
    # TRAINING
    # ========================================================

    best_val_f1 = -1.0

    epochs_without_improvement = 0


    for epoch in range(
        1,
        EPOCHS + 1
    ):


        # ====================================================
        # TRAIN
        # ====================================================

        (
            train_loss,
            train_acc,
            train_f1
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        (
            val_loss,
            val_acc,
            val_f1
        ) = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
        )


        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(
            val_f1
        )


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        # ====================================================
        # LOG
        # ====================================================

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


        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            epochs_without_improvement = 0


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "val_f1":
                        best_val_f1,

                    "epoch":
                        epoch,

                    "input_size":
                        INPUT_SIZE,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "num_layers":
                        NUM_LAYERS,

                    "num_classes":
                        NUM_CLASSES,

                    "dropout":
                        DROPOUT,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "seed":
                        SEED,
                },
                CHECKPOINT_PATH
            )


            print(
                f"  -> Saved best model "
                f"(Val F1="
                f"{best_val_f1:.4f})"
            )


        else:

            epochs_without_improvement += 1


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                f"Early stopping at "
                f"epoch {epoch}."
            )

            break


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print(
        "============================================"
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        f"Best validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        "============================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()