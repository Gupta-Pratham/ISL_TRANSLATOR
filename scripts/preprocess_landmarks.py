import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path(r".\data\processed\landmarks_clean")
OUTPUT_DIR = Path(r".\data\processed\landmarks_preprocessed")


# ============================================================
# REPRESENTATION
# ============================================================

# Landmark layout:
#
# 0  - 20  : Left hand
# 21 - 41  : Right hand
# 42 - 49  : Upper-body pose
#
# Pose indices correspond to:
# 42 = left shoulder
# 43 = right shoulder
# 44 = left elbow
# 45 = right elbow
# 46 = left wrist
# 47 = right wrist
# 48 = left hip
# 49 = right hip


LEFT_SHOULDER = 42
RIGHT_SHOULDER = 43

MIN_SHOULDER_DISTANCE = 1e-6


# ============================================================
# MISSING-VALUE INTERPOLATION
# ============================================================

def interpolate_sequence(sequence):
    """
    Fill NaN values independently for every landmark coordinate.

    Internal gaps:
        Linear interpolation.

    Boundary gaps:
        Nearest valid value.

    If a coordinate has no valid observation in the entire video,
    interpolation is impossible, so it is filled with 0.
    """

    result = sequence.copy()

    T, L, C = result.shape

    for landmark in range(L):

        for coord in range(C):

            values = result[:, landmark, coord]

            valid = ~np.isnan(values)

            # ------------------------------------------------
            # No valid observation exists for this coordinate
            # in the entire video.
            # ------------------------------------------------
            if valid.sum() == 0:
                result[:, landmark, coord] = 0.0
                continue

            valid_indices = np.where(valid)[0]

            # ------------------------------------------------
            # Fill internal and boundary gaps.
            # np.interp uses:
            # - linear interpolation internally
            # - first valid value on the left boundary
            # - last valid value on the right boundary
            # ------------------------------------------------
            result[:, landmark, coord] = np.interp(
                np.arange(T),
                valid_indices,
                values[valid]
            )

    return result


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sequence(sequence):
    """
    Normalize all landmarks using:

        Origin = shoulder midpoint
        Scale  = shoulder distance
    """

    result = sequence.copy()

    # --------------------------------------------------------
    # Get shoulders
    # --------------------------------------------------------

    left_shoulder = result[:, LEFT_SHOULDER, :]
    right_shoulder = result[:, RIGHT_SHOULDER, :]

    # --------------------------------------------------------
    # Shoulder midpoint
    # --------------------------------------------------------

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2.0

    # --------------------------------------------------------
    # Shoulder distance
    # --------------------------------------------------------

    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder,
        axis=1
    )

    # --------------------------------------------------------
    # Prevent division by extremely small values
    # --------------------------------------------------------

    shoulder_distance = np.maximum(
        shoulder_distance,
        MIN_SHOULDER_DISTANCE
    )

    # --------------------------------------------------------
    # Center landmarks around shoulder midpoint
    # --------------------------------------------------------

    result = (
        result
        - shoulder_center[:, None, :]
    )

    # --------------------------------------------------------
    # Scale using shoulder distance
    # --------------------------------------------------------

    result = (
        result
        / shoulder_distance[:, None, None]
    )

    return result


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_path):

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    sequence = np.load(
        input_path
    ).astype(np.float32)

    original_nan_count = np.isnan(sequence).sum()

    # --------------------------------------------------------
    # 1. Interpolation / missing-value handling
    # --------------------------------------------------------

    sequence = interpolate_sequence(
        sequence
    )

    remaining_nan_count = np.isnan(sequence).sum()

    # --------------------------------------------------------
    # 2. Normalization
    # --------------------------------------------------------

    sequence = normalize_sequence(
        sequence
    )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    final_nan_count = np.isnan(sequence).sum()
    final_inf_count = np.isinf(sequence).sum()

    if final_nan_count > 0 or final_inf_count > 0:

        print(
            f"WARNING: {input_path.name} still contains "
            f"NaN/Inf values!"
        )

        print(
            f"NaNs={final_nan_count}, "
            f"Inf={final_inf_count}"
        )

        return

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / input_path.name

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_path,
        sequence
    )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print(
        f"{input_path.name} | "
        f"shape={sequence.shape} | "
        f"NaNs before={original_nan_count} | "
        f"NaNs after interpolation={remaining_nan_count} | "
        f"min={sequence.min():.4f} | "
        f"max={sequence.max():.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find all landmark files
    # --------------------------------------------------------

    files = sorted(
        INPUT_DIR.rglob("*.npy")
    )

    if not files:

        print(
            "No .npy files found."
        )

        return

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        "============================================"
    )

    print(
        "ISL Landmark Preprocessing"
    )

    print(
        "============================================"
    )

    print(
        f"Input : {INPUT_DIR}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"Files : {len(files)}"
    )

    print(
        "============================================"
    )

    # --------------------------------------------------------
    # Process every file
    # --------------------------------------------------------

    for file in files:

        process_file(
            file
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print(
        "\nPreprocessing complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()