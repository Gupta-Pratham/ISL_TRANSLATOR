import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from src.models import TransformerClassifier


# ============================================================
# PATHS
# ============================================================

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

LANDMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_preprocessed"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "transformer_best.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_SIZE = 150

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256

NUM_CLASSES = 59

DROPOUT = 0.3

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

class_names_df = (
    pd.read_csv(TEST_CSV)
    [["class_id", "label"]]
    .drop_duplicates(
        subset="class_id"
    )
    .sort_values(
        "class_id"
    )
)

CLASS_NAMES = (
    class_names_df["label"]
    .tolist()
)


# ============================================================
# SAFETY CHECKS
# ============================================================

if len(CLASS_NAMES) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(CLASS_NAMES)} class names."
    )


ALL_CLASS_IDS = list(
    range(NUM_CLASSES)
)


# ============================================================
# HEADER
# ============================================================

print()
print("============================================")
print("ISL TRANSFORMER EVALUATION")
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
    f"Test CSV: {TEST_CSV}"
)

print(
    f"Landmark directory: {LANDMARK_DIR}"
)

print(
    f"Checkpoint: {CHECKPOINT_PATH}"
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

print("============================================")


# ============================================================
# CHECK REQUIRED PATHS
# ============================================================

if not TEST_CSV.exists():

    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

if not LANDMARK_DIR.exists():

    raise FileNotFoundError(
        "Landmark directory not found:\n"
        f"{LANDMARK_DIR}"
    )

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        "Transformer checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

test_dataset, test_loader = create_dataloader(
    TEST_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
    landmark_dir=LANDMARK_DIR
)

print()
print(
    f"Test samples: {len(test_dataset)}"
)


# ============================================================
# CREATE MODEL
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
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
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


model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for (
        sequences,
        labels,
        lengths,
        padding_mask
    ) in test_loader:

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
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            sequences,
            padding_mask
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = outputs.argmax(
            dim=1
        )


        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
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
# BASIC VALIDATION
# ============================================================

if len(all_labels) != len(
    all_predictions
):

    raise RuntimeError(
        "Number of labels and predictions "
        "do not match."
    )


if len(all_labels) == 0:

    raise RuntimeError(
        "No predictions were generated."
    )


if np.any(all_labels < 0) or np.any(
    all_labels >= NUM_CLASSES
):

    raise ValueError(
        "Invalid ground-truth class ID detected."
    )


if np.any(all_predictions < 0) or np.any(
    all_predictions >= NUM_CLASSES
):

    raise ValueError(
        "Invalid predicted class ID detected."
    )


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# ------------------------------------------------------------
# MACRO METRICS
#
# Explicitly evaluate all 59 classes.
# ------------------------------------------------------------

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=ALL_CLASS_IDS,
        average="macro",
        zero_division=0
    )
)


# ------------------------------------------------------------
# WEIGHTED METRICS
# ------------------------------------------------------------

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )
)


# ------------------------------------------------------------
# CLASS COVERAGE
# ------------------------------------------------------------

classes_present = np.unique(
    all_labels
)

num_classes_present = len(
    classes_present
)


# ============================================================
# TEST RESULTS
# ============================================================

print()
print("============================================")
print("TEST RESULTS")
print("============================================")

print(
    f"Test samples         : {len(test_dataset)}"
)

print(
    f"Accuracy             : {accuracy:.4f}"
)

print(
    f"Macro Precision      : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall         : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1             : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted Precision   : "
    f"{weighted_precision:.4f}"
)

print(
    f"Weighted Recall      : "
    f"{weighted_recall:.4f}"
)

print(
    f"Weighted F1          : "
    f"{weighted_f1:.4f}"
)

print(
    f"Classes in test set  : "
    f"{num_classes_present}/{NUM_CLASSES}"
)

print("============================================")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=ALL_CLASS_IDS,
    target_names=CLASS_NAMES,
    zero_division=0,
    digits=4,
)


print()
print("============================================")
print("PER-CLASS RESULTS")
print("============================================")

print(
    report
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_PATH = (
    RESULTS_DIR
    / "transformer_classification_report.txt"
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ISL TRANSFORMER CLASSIFICATION REPORT\n"
    )

    file.write(
        "============================================\n\n"
    )

    file.write(
        f"Test samples: "
        f"{len(test_dataset)}\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy:.4f}\n"
    )

    file.write(
        f"Macro Precision: "
        f"{macro_precision:.4f}\n"
    )

    file.write(
        f"Macro Recall: "
        f"{macro_recall:.4f}\n"
    )

    file.write(
        f"Macro F1: "
        f"{macro_f1:.4f}\n"
    )

    file.write(
        f"Weighted Precision: "
        f"{weighted_precision:.4f}\n"
    )

    file.write(
        f"Weighted Recall: "
        f"{weighted_recall:.4f}\n"
    )

    file.write(
        f"Weighted F1: "
        f"{weighted_f1:.4f}\n"
    )

    file.write(
        f"Classes in test set: "
        f"{num_classes_present}/{NUM_CLASSES}\n\n"
    )

    file.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=ALL_CLASS_IDS
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

CM_PATH = (
    RESULTS_DIR
    / "transformer_confusion_matrix.csv"
)


cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


cm_df.to_csv(
    CM_PATH
)


print()
print("============================================")
print("CONFUSION MATRIX")
print("============================================")

print(
    f"Saved to: {CM_PATH}"
)


# ============================================================
# INDIVIDUAL PREDICTIONS
# ============================================================

prediction_rows = []


for index, (
    true_label,
    predicted_label
) in enumerate(
    zip(
        all_labels,
        all_predictions
    )
):

    prediction_rows.append(
        {
            "sample_index": index + 1,

            "actual_class_id": int(
                true_label
            ),

            "actual_label": CLASS_NAMES[
                true_label
            ],

            "predicted_class_id": int(
                predicted_label
            ),

            "predicted_label": CLASS_NAMES[
                predicted_label
            ],

            "correct": bool(
                true_label == predicted_label
            ),
        }
    )


predictions_df = pd.DataFrame(
    prediction_rows
)


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "transformer_predictions.csv"
)


predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


print()
print("============================================")
print("PREDICTIONS")
print("============================================")

print(
    f"Saved to: {PREDICTIONS_PATH}"
)


# ============================================================
# FINISHED
# ============================================================

print()
print("============================================")
print("EVALUATION COMPLETE")
print("============================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {macro_precision:.4f}"
)

print(
    f"Recall    : {macro_recall:.4f}"
)

print(
    f"Macro F1  : {macro_f1:.4f}"
)

print("============================================")