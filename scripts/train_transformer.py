import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import create_dataloader
from src.models import TransformerClassifier


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 7
GRAD_CLIP = 1.0

INPUT_SIZE = 150
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 8
DROPOUT = 0.3

LANDMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_preprocessed"
)

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"

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
# Reproducibility
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Header
# ============================================================

print("============================================")
print("TRANSFORMER TRAINING")
print("============================================")
print(f"Device: {DEVICE}")
print(f"Input features: {INPUT_SIZE}")
print(f"d_model: {D_MODEL}")
print(f"Attention heads: {NHEAD}")
print(f"Transformer layers: {NUM_LAYERS}")
print(f"FFN dimension: {DIM_FEEDFORWARD}")
print(f"Classes: {NUM_CLASSES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("============================================")


# ============================================================
# Data
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

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")


# ============================================================
# Model
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

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Model parameters: {num_params:,}")


# ============================================================
# Class weights
# ============================================================

train_labels = np.array([
    train_dataset[i][1]
    for i in range(len(train_dataset))
])

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

class_weights = len(train_labels) / (
    NUM_CLASSES *
    np.maximum(class_counts, 1)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print(f"Class counts: {class_counts.tolist()}")
print(
    f"Class weights: "
    f"{class_weights.cpu().numpy()}"
)


# ============================================================
# Loss / Optimizer / Scheduler
# ============================================================

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
    factor=0.5,
    patience=3
)


# ============================================================
# Epoch function
# ============================================================

def run_epoch(model, loader, training=True):

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

        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(
                sequences,
                padding_mask
            )

            loss = criterion(
                outputs,
                labels
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                optimizer.step()

        total_loss += loss.item()

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

    avg_loss = (
        total_loss /
        len(loader)
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
# Training
# ============================================================

best_val_f1 = -1.0
epochs_without_improvement = 0

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    train_loss, train_acc, train_f1 = (
        run_epoch(
            model,
            train_loader,
            training=True
        )
    )

    val_loss, val_acc, val_f1 = (
        run_epoch(
            model,
            val_loader,
            training=False
        )
    )

    scheduler.step(val_f1)

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

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

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            f"Early stopping at "
            f"epoch {epoch}."
        )

        break


# ============================================================
# Finished
# ============================================================

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