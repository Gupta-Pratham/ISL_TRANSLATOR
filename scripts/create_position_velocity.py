from pathlib import Path

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

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
    / "landmarks_position_velocity"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_LANDMARKS = 50
COORDINATES = 3

POSITION_FEATURES = EXPECTED_LANDMARKS * COORDINATES
VELOCITY_FEATURES = POSITION_FEATURES

EXPECTED_OUTPUT_FEATURES = (
    POSITION_FEATURES + VELOCITY_FEATURES
)


# ============================================================
# CREATE FEATURES FOR ONE FILE
# ============================================================

def process_file(input_path):

    sequence = np.load(
        input_path
    ).astype(np.float32)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if sequence.ndim != 3:
        raise ValueError(
            f"Unexpected dimensions in {input_path.name}: "
            f"{sequence.shape}"
        )

    if sequence.shape[1:] != (
        EXPECTED_LANDMARKS,
        COORDINATES
    ):
        raise ValueError(
            f"Unexpected shape in {input_path.name}: "
            f"{sequence.shape}. "
            f"Expected (T, 50, 3)."
        )

    if np.isnan(sequence).any():
        raise ValueError(
            f"NaN values found in {input_path.name}"
        )

    if np.isinf(sequence).any():
        raise ValueError(
            f"Inf values found in {input_path.name}"
        )

    # --------------------------------------------------------
    # Position
    #
    # Shape:
    # (T, 50, 3)
    #       ↓
    # (T, 150)
    # --------------------------------------------------------

    positions = sequence.reshape(
        sequence.shape[0],
        -1
    )

    # --------------------------------------------------------
    # Velocity
    #
    # Velocity[t] = position[t] - position[t-1]
    #
    # First frame has no previous frame,
    # therefore velocity = 0.
    # --------------------------------------------------------

    velocity = np.zeros_like(
        positions,
        dtype=np.float32
    )

    if sequence.shape[0] > 1:

        velocity[1:] = (
            positions[1:]
            - positions[:-1]
        )

    # --------------------------------------------------------
    # Concatenate
    #
    # [position | velocity]
    #
    # (T, 150) + (T, 150)
    #       ↓
    # (T, 300)
    # --------------------------------------------------------

    features = np.concatenate(
        [
            positions,
            velocity
        ],
        axis=1
    ).astype(np.float32)

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if features.shape[1] != EXPECTED_OUTPUT_FEATURES:
        raise ValueError(
            f"Unexpected output feature count in "
            f"{input_path.name}: "
            f"{features.shape[1]}"
        )

    if np.isnan(features).any():
        raise ValueError(
            f"NaN values created in {input_path.name}"
        )

    if np.isinf(features).any():
        raise ValueError(
            f"Inf values created in {input_path.name}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / input_path.name
    )

    np.save(
        output_path,
        features
    )

    return features.shape


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("POSITION + VELOCITY FEATURE CREATION")
    print("=" * 70)

    print(f"Input directory : {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print(f"Position features : {POSITION_FEATURES}")
    print(f"Velocity features : {VELOCITY_FEATURES}")
    print(f"Total features    : {EXPECTED_OUTPUT_FEATURES}")
    print()

    # --------------------------------------------------------
    # Validate input directory
    # --------------------------------------------------------

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_DIR}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = sorted(
        INPUT_DIR.glob("*.npy")
    )

    if not files:

        raise FileNotFoundError(
            f"No .npy files found in:\n{INPUT_DIR}"
        )

    print(
        f"Input files found: {len(files)}"
    )
    print()

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    processed = 0

    for index, input_path in enumerate(
        files,
        start=1
    ):

        shape = process_file(
            input_path
        )

        processed += 1

        print(
            f"[{index}/{len(files)}] "
            f"{input_path.name} -> {shape}"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    output_files = sorted(
        OUTPUT_DIR.glob("*.npy")
    )

    print()
    print("=" * 70)
    print("FEATURE CREATION COMPLETE")
    print("=" * 70)

    print(
        f"Input files processed : {processed}"
    )

    print(
        f"Output files created  : {len(output_files)}"
    )

    if len(output_files) != len(files):

        raise RuntimeError(
            "Output file count does not match "
            "input file count."
        )

    # --------------------------------------------------------
    # Inspect one sample
    # --------------------------------------------------------

    sample_path = output_files[0]

    sample = np.load(
        sample_path
    )

    print()
    print(
        f"Sample file           : {sample_path.name}"
    )

    print(
        f"Sample shape          : {sample.shape}"
    )

    print(
        f"NaN count             : {np.isnan(sample).sum()}"
    )

    print(
        f"Inf count             : {np.isinf(sample).sum()}"
    )

    print()
    print(
        "Expected shape: "
        "(T, 300)"
    )

    print()
    print(
        "Original landmarks_preprocessed "
        "dataset was NOT modified."
    )


if __name__ == "__main__":
    main()