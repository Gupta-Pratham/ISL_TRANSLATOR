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

from src.dataset import create_dataloader


# ============================================================
# PATHS
# ============================================================

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

HANDS_ONLY_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_hands_only"
)


# ============================================================
# DATALOADER
# ============================================================

dataset, loader = create_dataloader(
    TEST_CSV,
    batch_size=4,
    shuffle=False,
    landmark_dir=HANDS_ONLY_DIR
)


# ============================================================
# GET ONE BATCH
# ============================================================

sequences, labels, lengths, padding_mask = next(
    iter(loader)
)


# ============================================================
# RESULTS
# ============================================================

print("============================================")
print("HANDS-ONLY DATASET TEST")
print("============================================")

print(
    f"Dataset size: {len(dataset)}"
)

print(
    f"Sequences shape: {sequences.shape}"
)

print(
    f"Labels shape: {labels.shape}"
)

print(
    f"Lengths: {lengths.tolist()}"
)

print(
    f"Padding mask shape: {padding_mask.shape}"
)

print(
    f"NaN: {torch.isnan(sequences).any().item()}"
)

print(
    f"Inf: {torch.isinf(sequences).any().item()}"
)

print("============================================")