import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from src.models import TCNClassifier


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"

LANDMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_masked"
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
    / "tcn_masked_best.pt"
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

INPUT_SIZE = 200
NUM_CLASSES = 59


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

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class MaskedISLDataset(Dataset):

    def __init__(self, csv_path):

        self.data = np.array(
            __import__("pandas").read_csv(csv_path)
            .to_dict("records"),
            dtype=object
        )

        self.path_map = {}

        files = LANDMARK_DIR.rglob("*.npy")

        for path in files:

            filename = path.name

            if filename in self.path_map:
                raise ValueError(
                    f"Duplicate landmark filename found: {filename}"
                )

            self.path_map[filename] = path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data[index]

        filename = f"{row['video_id']}.npy"

        if filename not in self.path_map:
            raise FileNotFoundError(
                f"Masked landmark file not found: {filename}"
            )

        path = self.path_map[filename]

        sequence = np.load(path).astype(np.float32)

        # ----------------------------------------------------
        # Validate shape
        # ----------------------------------------------------

        if sequence.ndim != 2:
            raise ValueError(
                f"Expected (T, 200), got {sequence.shape} "
                f"for {filename}"
            )

        if sequence.shape[1] != INPUT_SIZE:
            raise ValueError(
                f"Expected {INPUT_SIZE} features/frame, "
                f"got {sequence.shape[1]} for {filename}"
            )

        if np.isnan(sequence).any():
            raise ValueError(
                f"NaN detected in {filename}"
            )

        if np.isinf(sequence).any():
            raise ValueError(
                f"Inf detected in {filename}"
            )

        sequence = torch.from_numpy(sequence)

        label = int(row["class_id"])

        return sequence, label


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    sequences = [
        item[0]
        for item in batch
    ]

    labels = torch.tensor(
        [
            item[1]
            for item in batch
        ],
        dtype=torch.long
    )

    lengths = torch.tensor(
        [
            sequence.shape[0]
            for sequence in sequences
        ],
        dtype=torch.long
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    batch_size = padded_sequences.shape[0]
    max_length = padded_sequences.shape[1]

    padding_mask = torch.ones(
        batch_size,
        max_length,
        dtype=torch.bool
    )

    for i, length in enumerate(lengths):

        padding_mask[
            i,
            :length
        ] = False

    return (
        padded_sequences,
        labels,
        lengths,
        padding_mask
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = MaskedISLDataset(TRAIN_CSV)

val_dataset = MaskedISLDataset(VAL_CSV)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# HEADER
# ============================================================

print("============================================")
print("ISL TCN Missingness-Mask Training")
print("============================================")

print(f"Device: {DEVICE}")

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
    f"Landmark directory: {LANDMARK_DIR}"
)

print(
    f"Checkpoint: {BEST_MODEL_PATH}"
)

print("============================================")


# ============================================================
# DATASET CHECK
# ============================================================

print()

print(
    f"Training samples: {len(train_dataset)}"
)

print(
    f"Validation samples: {len(val_dataset)}"
)

sample_sequence, sample_label = train_dataset[0]

print(
    f"Sample shape: {tuple(sample_sequence.shape)}"
)

print(
    f"Sample label: {sample_label}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = []

for i in range(len(train_dataset)):

    _, label = train_dataset[i]

    train_labels.append(int(label))

train_labels = np.array(train_labels)

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
    input_size=INPUT_SIZE,
    hidden_size=128,
    num_classes=NUM_CLASSES,
    dropout=0.3
)

model = model.to(DEVICE)

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

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_losses = []

    train_predictions = []

    train_targets = []

    for batch in train_loader:

        sequences, labels, lengths, padding_mask = batch

        sequences = sequences.to(DEVICE)

        labels = labels.to(DEVICE)

        padding_mask = padding_mask.to(DEVICE)

        logits = model(
            sequences,
            padding_mask
        )

        loss = criterion(
            logits,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        optimizer.step()

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

            sequences = sequences.to(DEVICE)

            labels = labels.to(DEVICE)

            padding_mask = padding_mask.to(DEVICE)

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

    scheduler.step(val_f1)

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
            f"  -> Saved best model "
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