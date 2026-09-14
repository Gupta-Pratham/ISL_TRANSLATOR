import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import create_dataloader

TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"

POSE_ONLY_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_pose_only"
)

dataset, loader = create_dataloader(
    TEST_CSV,
    batch_size=4,
    shuffle=False,
    landmark_dir=POSE_ONLY_DIR
)

sequences, labels, lengths, padding_mask = next(iter(loader))

print("============================================")
print("POSE-ONLY DATASET TEST")
print("============================================")
print(f"Dataset size: {len(dataset)}")
print(f"Sequences shape: {sequences.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Lengths: {lengths.tolist()}")
print(f"Padding mask shape: {padding_mask.shape}")
print(f"NaN: {torch.isnan(sequences).any().item()}")
print(f"Inf: {torch.isinf(sequences).any().item()}")
print("============================================")