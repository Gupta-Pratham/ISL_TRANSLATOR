from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LANDMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_preprocessed"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "error_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LANDMARK LAYOUT
# ============================================================

# Our preprocessed representation:
#
# 0 - 20   : left hand
# 21 - 41  : right hand
# 42 - 49  : upper-body pose
#
# Each landmark has:
# x, y, z
#
# Therefore:
# sequence.shape = (T, 50, 3)


LEFT_HAND_START = 0
LEFT_HAND_END = 21

RIGHT_HAND_START = 21
RIGHT_HAND_END = 42

POSE_START = 42
POSE_END = 50


# ============================================================
# ERROR CASES
# ============================================================

ERROR_CASES = [
    {
        "video_id": "MVI_9313",
        "actual": "82. narrow",
        "predicted": "81. wide",
    },
    {
        "video_id": "MVI_5278",
        "actual": "81. wide",
        "predicted": "82. narrow",
    },
    {
        "video_id": "MVI_5200",
        "actual": "82. narrow",
        "predicted": "81. wide",
    },
]


# ============================================================
# MOTION CALCULATION
# ============================================================

def landmark_motion(sequence, start, end):
    """
    Calculate frame-to-frame motion magnitude
    for a selected landmark group.

    sequence:
        (T, landmarks, 3)

    returns:
        (T,) motion magnitude
    """

    group = sequence[:, start:end, :]

    # Difference between consecutive frames
    differences = np.diff(group, axis=0)

    # Euclidean movement of every landmark
    distances = np.linalg.norm(
        differences,
        axis=2
    )

    # Average movement across landmarks
    motion = distances.mean(axis=1)

    # Add zero for first frame
    motion = np.concatenate(
        [[0.0], motion]
    )

    return motion


# ============================================================
# HAND SPREAD
# ============================================================

def hand_center(sequence, start, end):
    """
    Calculate center of a hand for every frame.
    """

    hand = sequence[:, start:end, :]

    return hand.mean(axis=1)


def hand_spread(sequence, start, end):
    """
    Calculate average distance of hand landmarks
    from the hand center.
    """

    hand = sequence[:, start:end, :]

    center = hand.mean(axis=1, keepdims=True)

    distances = np.linalg.norm(
        hand - center,
        axis=2
    )

    return distances.mean(axis=1)


# ============================================================
# VISUALIZE ONE ERROR
# ============================================================

def visualize_case(case):

    video_id = case["video_id"]

    path = LANDMARK_DIR / f"{video_id}.npy"

    if not path.exists():
        print(
            f"WARNING: landmark file not found: {path}"
        )
        return

    sequence = np.load(path).astype(np.float32)

    if sequence.ndim != 3:
        raise ValueError(
            f"Unexpected shape for {video_id}: "
            f"{sequence.shape}"
        )

    print(
        f"{video_id}: shape={sequence.shape}"
    )

    # --------------------------------------------------------
    # Calculate motion
    # --------------------------------------------------------

    left_motion = landmark_motion(
        sequence,
        LEFT_HAND_START,
        LEFT_HAND_END
    )

    right_motion = landmark_motion(
        sequence,
        RIGHT_HAND_START,
        RIGHT_HAND_END
    )

    pose_motion = landmark_motion(
        sequence,
        POSE_START,
        POSE_END
    )

    # --------------------------------------------------------
    # Calculate hand spread
    # --------------------------------------------------------

    left_spread = hand_spread(
        sequence,
        LEFT_HAND_START,
        LEFT_HAND_END
    )

    right_spread = hand_spread(
        sequence,
        RIGHT_HAND_START,
        RIGHT_HAND_END
    )

    # --------------------------------------------------------
    # Time axis
    # --------------------------------------------------------

    frames = np.arange(
        sequence.shape[0]
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10)
    )

    fig.suptitle(
        f"TCN Error Analysis — {video_id}\n"
        f"Actual: {case['actual']}   |   "
        f"Predicted: {case['predicted']}",
        fontsize=14
    )

    # ========================================================
    # PLOT 1 — HAND MOTION
    # ========================================================

    axes[0].plot(
        frames,
        left_motion,
        label="Left-hand motion"
    )

    axes[0].plot(
        frames,
        right_motion,
        label="Right-hand motion"
    )

    axes[0].set_title(
        "Frame-to-frame hand motion"
    )

    axes[0].set_xlabel(
        "Frame"
    )

    axes[0].set_ylabel(
        "Mean landmark movement"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # ========================================================
    # PLOT 2 — POSE MOTION
    # ========================================================

    axes[1].plot(
        frames,
        pose_motion,
        label="Upper-body pose motion"
    )

    axes[1].set_title(
        "Upper-body motion"
    )

    axes[1].set_xlabel(
        "Frame"
    )

    axes[1].set_ylabel(
        "Mean landmark movement"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    # ========================================================
    # PLOT 3 — HAND SPREAD
    # ========================================================

    axes[2].plot(
        frames,
        left_spread,
        label="Left-hand spread"
    )

    axes[2].plot(
        frames,
        right_spread,
        label="Right-hand spread"
    )

    axes[2].set_title(
        "Hand configuration / spread"
    )

    axes[2].set_xlabel(
        "Frame"
    )

    axes[2].set_ylabel(
        "Mean distance from hand center"
    )

    axes[2].legend()

    axes[2].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{video_id}_error_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TCN ERROR VISUALIZATION")
    print("=" * 70)

    print(
        f"Landmark directory:\n{LANDMARK_DIR}"
    )

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    print()

    for case in ERROR_CASES:

        visualize_case(case)

        print()

    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()