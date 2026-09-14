import sys
from pathlib import Path

import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT
# ============================================================

from src.models import TCNClassifier


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

model = TCNClassifier(
    input_size=150,
    hidden_size=128,
    num_classes=8,
    dropout=0.3
)

model = model.to(
    DEVICE
)


# ============================================================
# DUMMY INPUT
# ============================================================

batch_size = 4
sequence_length = 70
input_size = 150

x = torch.randn(
    batch_size,
    sequence_length,
    input_size,
    device=DEVICE
)


# ============================================================
# PADDING MASK
# ============================================================

padding_mask = torch.zeros(
    batch_size,
    sequence_length,
    dtype=torch.bool,
    device=DEVICE
)


padding_mask[0, 60:] = True
padding_mask[1, 50:] = True
padding_mask[2, 65:] = True
padding_mask[3, 40:] = True


# ============================================================
# FORWARD PASS
# ============================================================

model.eval()

with torch.no_grad():

    output = model(
        x,
        padding_mask
    )


# ============================================================
# RESULTS
# ============================================================

print("============================================")
print("TCN MODEL TEST")
print("============================================")

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Input shape: {x.shape}"
)

print(
    f"Output shape: {output.shape}"
)

print(
    f"Parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

print(
    f"NaN: "
    f"{torch.isnan(output).any().item()}"
)

print(
    f"Inf: "
    f"{torch.isinf(output).any().item()}"
)

print("============================================")