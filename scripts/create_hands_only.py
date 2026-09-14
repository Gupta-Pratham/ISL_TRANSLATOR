from pathlib import Path
import numpy as np


# ============================================================
# PATHS
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
    / "landmarks_hands_only"
)


# ============================================================
# SETTINGS
# ============================================================

# First 42 landmarks = 21 left-hand + 21 right-hand
HAND_LANDMARKS = 42


# ============================================================
# CONVERT ONE FILE
# ============================================================

def process_file(input_path):

    sequence = np.load(
        input_path
    ).astype(np.float32)

    # --------------------------------------------------------
    # Expected:
    # (T, 50, 3)
    # --------------------------------------------------------

    if sequence.ndim != 3:
        raise ValueError(
            f"Unexpected shape: {sequence.shape}"
        )

    if sequence.shape[1] != 50:
        raise ValueError(
            f"Expected 50 landmarks, "
            f"got {sequence.shape[1]}"
        )

    # --------------------------------------------------------
    # Keep only both hands
    # --------------------------------------------------------

    hands_only = sequence[
        :, :HAND_LANDMARKS, :
    ]

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if np.isnan(hands_only).any():
        raise ValueError(
            f"NaNs found in {input_path.name}"
        )

    if np.isinf(hands_only).any():
        raise ValueError(
            f"Inf found in {input_path.name}"
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
        hands_only
    )

    print(
        f"{input_path.name} | "
        f"{sequence.shape} -> "
        f"{hands_only.shape}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        INPUT_DIR.glob("*.npy")
    )

    print("============================================")
    print("Creating Hands-Only Representation")
    print("============================================")
    print(
        f"Input files : {len(files)}"
    )
    print(
        f"Output dir  : {OUTPUT_DIR}"
    )
    print("============================================")

    for file in files:

        process_file(file)

    print()
    print("============================================")
    print("Conversion complete.")
    print("============================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()