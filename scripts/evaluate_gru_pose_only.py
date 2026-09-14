import sys
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import create_dataloader
from src.models import GRUClassifier


# ============================================================
# Paths
# ============================================================

TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"

LANDMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_pose_only"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "gru_pose_only_best.pt"
)


# ============================================================
# Configuration
# ============================================================

INPUT_SIZE = 24
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 8
DROPOUT = 0.3

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Class names
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
# Load dataset
# ============================================================

test_dataset, test_loader = create_dataloader(
    TEST_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
    landmark_dir=LANDMARK_DIR
)


# ============================================================
# Load model
# ============================================================

model = GRUClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Evaluation
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for sequences, labels, lengths, padding_mask in test_loader:

        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)

        outputs = model(
            sequences,
            padding_mask
        )

        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# Results
# ============================================================

print()
print("============================================")
print("HANDS-ONLY BiGRU TEST RESULTS")
print("============================================")

print(f"Test samples: {len(test_dataset)}")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall:    {recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")

print()
print("============================================")
print("CLASSIFICATION REPORT")
print("============================================")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("============================================")
print("CONFUSION MATRIX")
print("============================================")

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print(cm)

print("============================================")

print()
print("Individual predictions:")
print("--------------------------------------------")

for i, (true_label, pred_label) in enumerate(
    zip(all_labels, all_predictions)
):

    true_name = CLASS_NAMES[true_label]
    pred_name = CLASS_NAMES[pred_label]

    status = "✓" if true_label == pred_label else "✗"

    print(
        f"{i+1:02d}. "
        f"True: {true_name:<10} "
        f"Predicted: {pred_name:<10} "
        f"{status}"
    )

print("============================================")