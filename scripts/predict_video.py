import sys
from pathlib import Path

import numpy as np
import torch

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from src.models import TCNClassifier


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "tcn_baseline_best.pt"
)

LANDMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_preprocessed"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

INPUT_SIZE = 150
NUM_CLASSES = 59

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD LABEL MAP
# ============================================================

import pandas as pd

test_df = pd.read_csv(TEST_CSV)

label_map = (
    test_df[["class_id", "label"]]
    .drop_duplicates()
    .set_index("class_id")["label"]
    .to_dict()
)


# ============================================================
# LOAD MODEL
# ============================================================

model = TCNClassifier(
    input_size=INPUT_SIZE,
    hidden_size=128,
    num_classes=NUM_CLASSES,
    dropout=0.3
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# PREDICT
# ============================================================

def predict(video_id):

    filename = f"{video_id}.npy"

    landmark_path = LANDMARK_DIR / filename

    if not landmark_path.exists():
        raise FileNotFoundError(
            f"Landmark file not found:\n{landmark_path}"
        )

    sequence = np.load(
        landmark_path
    ).astype(np.float32)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if sequence.ndim != 3:
        raise ValueError(
            f"Expected (T, 50, 3), got {sequence.shape}"
        )

    if sequence.shape[1:] != (50, 3):
        raise ValueError(
            f"Expected (T, 50, 3), got {sequence.shape}"
        )

    if np.isnan(sequence).any():
        raise ValueError(
            "NaN detected in landmark sequence."
        )

    if np.isinf(sequence).any():
        raise ValueError(
            "Inf detected in landmark sequence."
        )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    sequence = sequence.reshape(
        sequence.shape[0],
        -1
    )

    if sequence.shape[1] != INPUT_SIZE:
        raise ValueError(
            f"Expected {INPUT_SIZE} features/frame, "
            f"got {sequence.shape[1]}"
        )

    # --------------------------------------------------------
    # Tensor
    #
    # Shape: (1, T, 150)
    # --------------------------------------------------------

    x = torch.from_numpy(
        sequence
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Padding mask
    #
    # No padding because this is one complete sequence.
    # False = valid frame.
    # --------------------------------------------------------

    padding_mask = torch.zeros(
        1,
        sequence.shape[0],
        dtype=torch.bool,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            x,
            padding_mask
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    predicted_label = label_map.get(
        predicted_class,
        f"class_{predicted_class}"
    )

    return (
        predicted_class,
        predicted_label,
        confidence
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python scripts\\predict_video.py MVI_9531"
        )

        sys.exit(1)

    video_id = sys.argv[1]

    print("=" * 60)
    print("ISL TCN VIDEO PREDICTION")
    print("=" * 60)

    print(f"Device     : {DEVICE}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Video ID   : {video_id}")
    print()

    predicted_class, predicted_label, confidence = predict(
        video_id
    )

    print("-" * 60)

    print(
        f"Predicted class      : {predicted_class}"
    )

    print(
        f"Predicted ISL word   : {predicted_label}"
    )

    print(
        f"Confidence            : {confidence:.4f}"
    )

    print("-" * 60)