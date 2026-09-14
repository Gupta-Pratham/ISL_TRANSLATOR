import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

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

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "lstm_baseline_best.pt"
)

BATCH_SIZE = 8


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "loud",
    "quiet",
    "happy",
    "sad",
    "Beautiful",
    "Ugly",
    "Deaf",
    "Blind",
]


# ============================================================
# HEADER
# ============================================================

print("============================================")
print("ISL LSTM Evaluation")
print("============================================")

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Test CSV: {TEST_CSV}"
)

print(
    f"Model: {MODEL_PATH}"
)

print("============================================")


# ============================================================
# LOAD TEST DATA
# ============================================================

test_dataset, test_loader = create_dataloader(
    TEST_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(
    f"Test samples: {len(test_dataset)}"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = LSTMClassifier(
    input_size=150,
    hidden_size=128,
    num_layers=2,
    num_classes=8,
    dropout=0.3,
)


# ============================================================
# LOAD TRAINED CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.to(DEVICE)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_labels = []


with torch.no_grad():

    for batch in test_loader:

        # ----------------------------------------------------
        # create_dataloader returns:
        #
        # padded_sequences
        # labels
        # lengths
        # padding_mask
        #
        # Therefore batch is a tuple, not a dictionary.
        # ----------------------------------------------------

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
        # LSTMClassifier.forward() expects:
        #
        # model(x, padding_mask)
        # ----------------------------------------------------

        logits = model(
            sequences,
            padding_mask
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )
)


# ============================================================
# TEST RESULTS
# ============================================================

print()
print("============================================")
print("TEST RESULTS")
print("============================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"Macro F1 : {f1:.4f}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print("============================================")
print("PER-CLASS RESULTS")
print("============================================")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES)))
)


print(
    "============================================"
)

print(
    "CONFUSION MATRIX"
)

print(
    "============================================"
)

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()


print(
    f"{'':>12}",
    end=""
)

for name in CLASS_NAMES:

    print(
        f"{name:>10}",
        end=""
    )

print()


for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:>12}",
        end=""
    )

    for value in row:

        print(
            f"{value:>10}",
            end=""
        )

    print()


# ============================================================
# INDIVIDUAL PREDICTIONS
# ============================================================

print()
print("============================================")
print("PREDICTIONS")
print("============================================")

for index, (
    true_label,
    predicted_label
) in enumerate(
    zip(
        all_labels,
        all_predictions
    )
):

    status = (
        "CORRECT"
        if true_label == predicted_label
        else "WRONG"
    )

    print(
        f"{index + 1:02d}. "
        f"Actual={CLASS_NAMES[true_label]:<10} "
        f"Predicted={CLASS_NAMES[predicted_label]:<10} "
        f"{status}"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("============================================")
print("Evaluation complete.")
print("============================================")