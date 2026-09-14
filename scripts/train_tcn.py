import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


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

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "tcn_baseline_best.pt"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

SEED = 42

BATCH_SIZE = 8

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 7

GRADIENT_CLIP = 1.0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    SEED
)

np.random.seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# HEADER
# ============================================================

print("============================================")
print("ISL TCN Baseline Training")
print("============================================")

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Train CSV: {TRAIN_CSV}"
)

print(
    f"Validation CSV: {VAL_CSV}"
)

print(
    f"Checkpoint: {BEST_MODEL_PATH}"
)

print("============================================")


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset, train_loader = create_dataloader(
    TRAIN_CSV,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_dataset, val_loader = create_dataloader(
    VAL_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


print(
    f"Training samples: {len(train_dataset)}"
)

print(
    f"Validation samples: {len(val_dataset)}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = []

for i in range(
    len(train_dataset)
):

    _, label = train_dataset[i]

    train_labels.append(
        int(label)
    )


train_labels = np.array(
    train_labels
)

num_classes = 8

class_counts = np.bincount(
    train_labels,
    minlength=num_classes
)

class_weights = (
    len(train_labels)
    / (
        num_classes
        * np.maximum(
            class_counts,
            1
        )
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


print()

print(
    "Training class counts:",
    class_counts
)

print(
    "Class weights:",
    class_weights.cpu().numpy()
)


# ============================================================
# MODEL
# ============================================================

model = TCNClassifier(
    input_size=150,
    hidden_size=128,
    num_classes=8,
    dropout=0.3
)

model = model.to(
    DEVICE
)


print()

print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ============================================================
# TRACKING
# ============================================================

best_val_f1 = -1.0

epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_losses = []

    train_predictions = []

    train_targets = []


    for batch in train_loader:

        sequences, labels, lengths, padding_mask = batch

        sequences = sequences.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        padding_mask = padding_mask.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            sequences,
            padding_mask
        )

        loss = criterion(
            logits,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )


        optimizer.step()


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        train_losses.append(
            loss.item()
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        train_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


    # --------------------------------------------------------
    # TRAIN METRICS
    # --------------------------------------------------------

    train_loss = np.mean(
        train_losses
    )

    train_accuracy = accuracy_score(
        train_targets,
        train_predictions
    )

    train_f1 = f1_score(
        train_targets,
        train_predictions,
        average="macro",
        zero_division=0
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_losses = []

    val_predictions = []

    val_targets = []


    with torch.no_grad():

        for batch in val_loader:

            sequences, labels, lengths, padding_mask = batch

            sequences = sequences.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            padding_mask = padding_mask.to(
                DEVICE
            )


            logits = model(
                sequences,
                padding_mask
            )


            loss = criterion(
                logits,
                labels
            )


            val_losses.append(
                loss.item()
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_targets.extend(
                labels.cpu().numpy()
            )


    # --------------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------------

    val_loss = np.mean(
        val_losses
    )

    val_accuracy = accuracy_score(
        val_targets,
        val_predictions
    )

    val_f1 = f1_score(
        val_targets,
        val_predictions,
        average="macro",
        zero_division=0
    )


    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"LR={current_lr:.6f} | "
        f"Train Loss={train_loss:.4f} | "
        f"Train Acc={train_accuracy:.4f} | "
        f"Train F1={train_f1:.4f} | "
        f"Val Loss={val_loss:.4f} | "
        f"Val Acc={val_accuracy:.4f} | "
        f"Val F1={val_f1:.4f}"
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        val_f1
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "val_accuracy": val_accuracy,
            },
            BEST_MODEL_PATH
        )

        print(
            f"  → Saved best model "
            f"(Val F1={val_f1:.4f})"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()

        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# FINISHED
# ============================================================

print()

print(
    "============================================"
)

print(
    "Training complete"
)

print(
    "============================================"
)

print(
    f"Best validation F1: "
    f"{best_val_f1:.4f}"
)

print(
    "Best model saved to:"
)

print(
    BEST_MODEL_PATH
)