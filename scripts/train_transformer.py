import sys
from pathlib import Path

import numpy as np
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
from src.models import TransformerClassifier


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 8
NUM_EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 7
GRAD_CLIP = 1.0

# ------------------------------------------------------------
# Transformer architecture
# ------------------------------------------------------------

INPUT_SIZE = 150

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256

NUM_CLASSES = 59

DROPOUT = 0.3


# ============================================================
# DATA PATHS
# ============================================================

LANDMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_preprocessed"
)

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


# ============================================================
# CHECKPOINT PATH
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
    / "transformer_best.pt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

np.random.seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print()
print("============================================")
print("TRANSFORMER TRAINING")
print("============================================")

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Input features: {INPUT_SIZE}"
)

print(
    f"d_model: {D_MODEL}"
)

print(
    f"Attention heads: {NHEAD}"
)

print(
    f"Transformer layers: {NUM_LAYERS}"
)

print(
    f"FFN dimension: {DIM_FEEDFORWARD}"
)

print(
    f"Classes: {NUM_CLASSES}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Epochs: {NUM_EPOCHS}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Weight decay: {WEIGHT_DECAY}"
)

print(
    f"Dropout: {DROPOUT}"
)

print("============================================")


# ============================================================
# CHECK REQUIRED PATHS
# ============================================================

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
        f"Landmark directory not found:\n"
        f"{LANDMARK_DIR}"
    )


# ============================================================
# LOAD DATA
# ============================================================

train_dataset, train_loader = create_dataloader(
    TRAIN_CSV,
    batch_size=BATCH_SIZE,
    shuffle=True,
    landmark_dir=LANDMARK_DIR
)

val_dataset, val_loader = create_dataloader(
    VAL_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
    landmark_dir=LANDMARK_DIR
)


print()
print(
    f"Train samples: {len(train_dataset)}"
)

print(
    f"Val samples: {len(val_dataset)}"
)


# ============================================================
# VALIDATE DATASET LABELS
# ============================================================

train_labels = np.array([
    train_dataset[i][1]
    for i in range(len(train_dataset))
])

val_labels = np.array([
    val_dataset[i][1]
    for i in range(len(val_dataset))
])


if len(train_labels) == 0:

    raise RuntimeError(
        "Training dataset is empty."
    )

if len(val_labels) == 0:

    raise RuntimeError(
        "Validation dataset is empty."
    )


if np.any(train_labels < 0) or np.any(
    train_labels >= NUM_CLASSES
):

    raise ValueError(
        "Training dataset contains "
        "invalid class IDs."
    )


if np.any(val_labels < 0) or np.any(
    val_labels >= NUM_CLASSES
):

    raise ValueError(
        "Validation dataset contains "
        "invalid class IDs."
    )


# ============================================================
# MODEL
# ============================================================

model = TransformerClassifier(
    input_size=INPUT_SIZE,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(DEVICE)


# ============================================================
# MODEL PARAMETERS
# ============================================================

num_params = sum(
    p.numel()
    for p in model.parameters()
)


print()
print(
    f"Model parameters: {num_params:,}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)


class_weights = (
    len(train_labels)
    / (
        NUM_CLASSES
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
    f"Class counts: "
    f"{class_counts.tolist()}"
)

print(
    f"Class weights: "
    f"{class_weights.cpu().numpy()}"
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

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING-RATE SCHEDULER
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ============================================================
# EPOCH FUNCTION
# ============================================================

def run_epoch(
    model,
    loader,
    training=True
):

    if training:

        model.train()

    else:

        model.eval()


    total_loss = 0.0

    all_labels = []
    all_predictions = []


    for (
        sequences,
        labels,
        lengths,
        padding_mask
    ) in loader:

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
        # Clear gradients
        # ----------------------------------------------------

        if training:

            optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.set_grad_enabled(
            training
        ):

            outputs = model(
                sequences,
                padding_mask
            )


            loss = criterion(
                outputs,
                labels
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            if training:

                loss.backward()


                # --------------------------------------------
                # Gradient clipping
                # --------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )


                optimizer.step()


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        total_loss += loss.item()


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = outputs.argmax(
            dim=1
        )


        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


    # ========================================================
    # EPOCH METRICS
    # ========================================================

    avg_loss = (
        total_loss
        / len(loader)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    # --------------------------------------------------------
    # Explicitly average over all 59 classes.
    # This keeps the training metric definition consistent
    # with our corrected TCN evaluation.
    # --------------------------------------------------------

    _, _, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
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
# TRAINING
# ============================================================

best_val_f1 = -1.0

epochs_without_improvement = 0


for epoch in range(
    1,
    NUM_EPOCHS + 1
):


    # ========================================================
    # TRAIN
    # ========================================================

    train_loss, train_acc, train_f1 = (
        run_epoch(
            model,
            train_loader,
            training=True
        )
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    val_loss, val_acc, val_f1 = (
        run_epoch(
            model,
            val_loader,
            training=False
        )
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_f1
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ========================================================
    # LOG
    # ========================================================

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


    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "val_f1":
                    best_val_f1,

                "input_size":
                    INPUT_SIZE,

                "d_model":
                    D_MODEL,

                "nhead":
                    NHEAD,

                "num_layers":
                    NUM_LAYERS,

                "dim_feedforward":
                    DIM_FEEDFORWARD,

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
            f"  -> Saved best checkpoint "
            f"(Val F1 = "
            f"{best_val_f1:.4f})"
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

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


# ============================================================
# FINISHED
# ============================================================

print()
print("============================================")
print("TRAINING COMPLETE")
print("============================================")

print(
    f"Best validation F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Checkpoint: "
    f"{CHECKPOINT_PATH}"
)

print("============================================")