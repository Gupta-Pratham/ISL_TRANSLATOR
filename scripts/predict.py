from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models import TransformerClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "transformer_best.pt"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"


# --------------------------------------------------
# Class mapping
# --------------------------------------------------

CLASS_NAMES = {
    0: "1. loud",
    1: "2. quiet",
    2: "3. happy",
    3: "4. sad",
    4: "5. Beautiful",
    5: "6. Ugly",
    6: "7. Deaf",
    7: "8. Blind",
}


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model = TransformerClassifier(
    input_size=checkpoint["input_size"],
    d_model=checkpoint["d_model"],
    nhead=checkpoint["nhead"],
    num_layers=checkpoint["num_layers"],
    dim_feedforward=checkpoint["dim_feedforward"],
    num_classes=checkpoint["num_classes"],
    dropout=checkpoint["dropout"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# --------------------------------------------------
# Select a test sample
# --------------------------------------------------

import pandas as pd

df = pd.read_csv(TEST_CSV)

sample = df.iloc[0]

video_id = sample["video_id"]
true_label = int(sample["class_id"])

landmark_path = PROJECT_ROOT / "data" / "processed" / "landmarks_preprocessed" / f"{video_id}.npy"

print("=" * 60)
print("ISL TRANSFORMER — OFFLINE INFERENCE")
print("=" * 60)

print(f"Device: {device}")
print(f"Video ID: {video_id}")
print(f"Landmark file: {landmark_path}")
print(f"True label: {CLASS_NAMES[true_label]}")


# --------------------------------------------------
# Load landmarks
# --------------------------------------------------

sequence = np.load(landmark_path).astype(np.float32)

if sequence.ndim != 3:
    raise ValueError(
        f"Expected shape (T, 50, 3), got {sequence.shape}"
    )

T, num_landmarks, coordinates = sequence.shape

if num_landmarks != 50 or coordinates != 3:
    raise ValueError(
        f"Expected (T, 50, 3), got {sequence.shape}"
    )

if not np.isfinite(sequence).all():
    raise ValueError("Sequence contains NaN or Inf values.")


# --------------------------------------------------
# Convert to tensor
# --------------------------------------------------

sequence = sequence.reshape(T, 150)

x = torch.from_numpy(sequence).unsqueeze(0).to(device)

lengths = torch.tensor([T], device=device)

padding_mask = torch.zeros(
    (1, T),
    dtype=torch.bool,
    device=device
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():
    logits = model(
        x,
        padding_mask=padding_mask
    )

    probabilities = torch.softmax(logits, dim=1)

    predicted_class = int(torch.argmax(probabilities, dim=1).item())

    confidence = float(probabilities[0, predicted_class].item())


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print(f"Frames: {T}")
print(f"Predicted: {CLASS_NAMES[predicted_class]}")
print(f"Confidence: {confidence * 100:.2f}%")
print(f"Correct: {predicted_class == true_label}")

print("=" * 60)