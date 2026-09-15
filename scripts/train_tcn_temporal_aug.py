from pathlib import Path
import sys
import random

# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models import TCNClassifier


# ============================================================
# PATHS
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
# CONFIGURATION
# ============================================================

INPUT_SIZE = 150
NUM_CLASSES = 59

BATCH_SIZE = 8

NUM_EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

LR_FACTOR = 0.5
LR_PATIENCE = 3

EARLY_STOP_PATIENCE = 7

GRAD_CLIP = 1.0

SEED = 42


# ============================================================
# TEMPORAL AUGMENTATION
# ============================================================

AUGMENT_PROBABILITY = 0.5

MIN_TIME_SCALE = 0.8
MAX_TIME_SCALE = 1.2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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


set_seed(SEED)


# ============================================================
# TEMPORAL RESAMPLING
# ============================================================

def temporal_resample(sequence):
    """
    Randomly speed up or slow down a sequence.

    Input:
        sequence: (T, F)

    Output:
        augmented sequence: (T_new, F)

    The feature dimension remains unchanged.
    """

    if sequence.shape[0] < 3:
        return sequence

    time_scale = np.random.uniform(
        MIN_TIME_SCALE,
        MAX_TIME_SCALE
    )

    original_length = sequence.shape[0]

    new_length = int(
        round(
            original_length
            * time_scale
        )
    )

    new_length = max(
        3,
        new_length
    )

    old_positions = np.linspace(
        0.0,
        1.0,
        original_length
    )

    new_positions = np.linspace(
        0.0,
        1.0,
        new_length
    )

    augmented = np.empty(
        (
            new_length,
            sequence.shape[1]
        ),
        dtype=np.float32
    )

    for feature_index in range(
        sequence.shape[1]
    ):

        augmented[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index]
        )

    return augmented


# ============================================================
# DATASET
# ============================================================

class TemporalAugmentedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_path,
        landmark_dir,
        training=False
    ):

        self.data = pd.read_csv(
            csv_path
        )

        self.landmark_dir = Path(
            landmark_dir
        )

        self.training = training

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        filename = (
            f"{row['video_id']}.npy"
        )

        path = (
            self.landmark_dir
            / filename
        )

        sequence = np.load(
        path
        ).astype(np.float32)

        # --------------------------------------------------------
        # Original landmark representation:
        # (T, 50, 3)
        # --------------------------------------------------------

        if sequence.ndim != 3:

            raise ValueError(
                f"Expected (T, 50, 3), "
                f"got {sequence.shape} "
                f"for {filename}"
            )

        if sequence.shape[1:] != (50, 3):

            raise ValueError(
                f"Expected (T, 50, 3), "
                f"got {sequence.shape} "
                f"for {filename}"
            )

        if np.isnan(sequence).any():

            raise ValueError(
                f"NaN detected in {filename}"
            )

        if np.isinf(sequence).any():

            raise ValueError(
                f"Inf detected in {filename}"
            )

        # --------------------------------------------------------
        # Flatten landmarks:
        #
        # (T, 50, 3)
        #      ↓
        # (T, 150)
        # --------------------------------------------------------

        sequence = sequence.reshape(
            sequence.shape[0],
            -1
        )

        if sequence.shape[1] != INPUT_SIZE:

            raise ValueError(
                f"Expected {INPUT_SIZE} "
                f"features/frame after reshaping, "
                f"got {sequence.shape[1]} "
                f"for {filename}"
            )

        # ----------------------------------------------------
        # Apply augmentation ONLY during training
        # ----------------------------------------------------

        if self.training:

            if np.random.random() < AUGMENT_PROBABILITY:

                sequence = temporal_resample(
                    sequence
                )

        sequence = torch.from_numpy(
            sequence
        )

        label = int(
            row["class_id"]
        )

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

    padded_sequences = (
        torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0.0
        )
    )

    batch_size = (
        padded_sequences.shape[0]
    )

    max_length = (
        padded_sequences.shape[1]
    )

    padding_mask = torch.ones(
        batch_size,
        max_length,
        dtype=torch.bool
    )

    for i, length in enumerate(
        lengths
    ):

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
# DATALOADER
# ============================================================

def create_loader(
    csv_path,
    landmark_dir,
    batch_size,
    shuffle,
    training
):

    dataset = TemporalAugmentedDataset(
        csv_path,
        landmark_dir,
        training=training
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return dataset, loader


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    csv_path
):

    df = pd.read_csv(
        csv_path
    )

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

    for class_id in range(
        NUM_CLASSES
    ):

        class_count = counts.get(
            class_id,
            0
        )

        if class_count > 0:

            weights[class_id] = (
                total_samples
                / (
                    NUM_CLASSES
                    * class_count
                )
            )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# MACRO F1
# ============================================================

def calculate_macro_f1(
    y_true,
    y_pred
):

    f1_scores = []

    for class_id in range(
        NUM_CLASSES
    ):

        tp = np.sum(
            (y_true == class_id)
            & (y_pred == class_id)
        )

        fp = np.sum(
            (y_true != class_id)
            & (y_pred == class_id)
        )

        fn = np.sum(
            (y_true == class_id)
            & (y_pred != class_id)
        )

        precision_denominator = (
            tp + fp
        )

        recall_denominator = (
            tp + fn
        )

        precision = (
            tp / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            tp / recall_denominator
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

        f1_scores.append(
            f1
        )

    return float(
        np.mean(f1_scores)
    )


# ============================================================
# TRAIN
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

    for (
        sequences,
        labels,
        lengths,
        padding_mask
    ) in loader:

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

        predictions = (
            outputs.argmax(
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

    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = np.mean(
        all_predictions
        == all_labels
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

        for (
            sequences,
            labels,
            lengths,
            padding_mask
        ) in loader:

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

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = np.mean(
        all_predictions
        == all_labels
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
    print("TCN + TEMPORAL AUGMENTATION")
    print("=" * 70)

    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Input features      : {INPUT_SIZE}"
    )

    print(
        f"Classes             : {NUM_CLASSES}"
    )

    print(
        f"Batch size          : {BATCH_SIZE}"
    )

    print(
        f"Augmentation prob.  : "
        f"{AUGMENT_PROBABILITY}"
    )

    print(
        f"Time scale range    : "
        f"{MIN_TIME_SCALE} - "
        f"{MAX_TIME_SCALE}"
    )

    print()

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create loaders
    # --------------------------------------------------------

    train_dataset, train_loader = create_loader(
        TRAIN_CSV,
        LANDMARK_DIR,
        BATCH_SIZE,
        shuffle=True,
        training=True
    )

    val_dataset, val_loader = create_loader(
        VAL_CSV,
        LANDMARK_DIR,
        BATCH_SIZE,
        shuffle=False,
        training=False
    )

    print(
        f"Training samples   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # Verify original representation
    # --------------------------------------------------------

    sample_sequence, sample_label = (
        train_dataset[0]
    )

    print(
        f"Sample shape       : "
        f"{tuple(sample_sequence.shape)}"
    )

    if sample_sequence.shape[1] != INPUT_SIZE:

        raise ValueError(
            "Feature dimension is incorrect."
        )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            TRAIN_CSV
        ).to(DEVICE)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TCNClassifier(
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters         : "
        f"{parameter_count:,}"
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

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_f1 = -1.0
    best_epoch = 0

    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "tcn_temporal_aug_best.pt"
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        (
            train_loss,
            train_acc,
            train_f1
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        (
            val_loss,
            val_acc,
            val_f1
        ) = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_f1
        )

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

        # ----------------------------------------------------
        # Save best model
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
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                    "augmentation": (
                        "temporal_resampling"
                    ),
                    "augmentation_probability": (
                        AUGMENT_PROBABILITY
                    ),
                    "min_time_scale": (
                        MIN_TIME_SCALE
                    ),
                    "max_time_scale": (
                        MAX_TIME_SCALE
                    ),
                },
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print(
                f"  -> New best validation "
                f"F1: {best_val_f1:.4f}"
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

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

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
        "Baseline TCN checkpoint "
        "was not modified."
    )


if __name__ == "__main__":
    main()