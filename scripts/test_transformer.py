import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models import TransformerClassifier


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4
TIME_STEPS = 65
INPUT_SIZE = 150
NUM_CLASSES = 8


model = TransformerClassifier(
    input_size=INPUT_SIZE,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    num_classes=NUM_CLASSES,
    dropout=0.3,
).to(DEVICE)


x = torch.randn(
    BATCH_SIZE,
    TIME_STEPS,
    INPUT_SIZE,
).to(DEVICE)


padding_mask = torch.zeros(
    BATCH_SIZE,
    TIME_STEPS,
    dtype=torch.bool,
).to(DEVICE)

# Simulate padding in the last 10 frames
padding_mask[:, -10:] = True


with torch.no_grad():

    output = model(
        x,
        padding_mask
    )


num_params = sum(
    p.numel()
    for p in model.parameters()
)


print("============================================")
print("TRANSFORMER MODEL TEST")
print("============================================")
print(f"Device: {DEVICE}")
print(f"Input shape: {x.shape}")
print(f"Padding mask shape: {padding_mask.shape}")
print(f"Output shape: {output.shape}")
print(f"Parameters: {num_params:,}")
print(f"NaN: {torch.isnan(output).any().item()}")
print(f"Inf: {torch.isinf(output).any().item()}")
print("============================================")