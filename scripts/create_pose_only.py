from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_preprocessed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_pose_only"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSE_START = 42
POSE_END = 50


files = sorted(INPUT_DIR.glob("*.npy"))

print("============================================")
print("CREATING POSE-ONLY DATASET")
print("============================================")
print(f"Input files: {len(files)}")
print(f"Output directory: {OUTPUT_DIR}")
print()

for file_path in files:

    sequence = np.load(file_path)

    if sequence.ndim != 3:
        raise ValueError(
            f"{file_path.name}: expected 3D array, "
            f"got shape {sequence.shape}"
        )

    if sequence.shape[1:] != (50, 3):
        raise ValueError(
            f"{file_path.name}: expected "
            f"(T, 50, 3), got {sequence.shape}"
        )

    pose_only = sequence[:, POSE_START:POSE_END, :]

    output_path = OUTPUT_DIR / file_path.name

    np.save(output_path, pose_only)

    print(
        f"{file_path.name}: "
        f"{sequence.shape} -> {pose_only.shape}"
    )


print()
print("============================================")
print("CONVERSION COMPLETE")
print("============================================")