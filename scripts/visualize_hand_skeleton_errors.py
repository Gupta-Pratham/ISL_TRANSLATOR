from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LANDMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landmarks_preprocessed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "error_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HAND CONNECTIONS
# MediaPipe hand landmark topology
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky

    # Palm
    (5, 9),
    (9, 13),
    (13, 17),
]


# ============================================================
# DATA LAYOUT
# ============================================================

LEFT_HAND_START = 0
LEFT_HAND_END = 21

RIGHT_HAND_START = 21
RIGHT_HAND_END = 42


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
# FRAME SELECTION
# ============================================================

def select_frames(num_frames):
    """
    Select five approximately equally spaced frames:
    beginning, 25%, 50%, 75%, end.
    """

    if num_frames < 5:
        return np.arange(num_frames)

    indices = np.linspace(
        0,
        num_frames - 1,
        5
    ).round().astype(int)

    return indices


# ============================================================
# DRAW ONE HAND
# ============================================================

def draw_hand(ax, hand, title):

    # MediaPipe x/y coordinates
    x = hand[:, 0]
    y = hand[:, 1]

    # Draw connections
    for start, end in HAND_CONNECTIONS:

        ax.plot(
            [x[start], x[end]],
            [y[start], y[end]],
            linewidth=1.5
        )

    # Draw landmarks
    ax.scatter(
        x,
        y,
        s=25
    )

    # Label landmark IDs
    for i in range(len(hand)):

        ax.text(
            x[i],
            y[i],
            str(i),
            fontsize=6
        )

    ax.set_title(title)

    # Image coordinates have y increasing downward
    ax.invert_yaxis()

    ax.set_aspect("equal", adjustable="box")

    ax.grid(
        alpha=0.25
    )


# ============================================================
# NORMALIZE HAND FOR VISUALIZATION
# ============================================================

def normalize_hand(hand):

    hand = hand.copy()

    # Wrist as origin
    wrist = hand[0].copy()

    hand[:, :3] -= wrist

    # Scale using maximum distance from wrist
    distances = np.linalg.norm(
        hand[:, :3],
        axis=1
    )

    scale = distances.max()

    if scale > 1e-8:
        hand[:, :3] /= scale

    return hand


# ============================================================
# VISUALIZE ONE VIDEO
# ============================================================

def visualize_case(case):

    video_id = case["video_id"]

    path = LANDMARK_DIR / f"{video_id}.npy"

    if not path.exists():

        print(
            f"WARNING: file not found:\n{path}"
        )

        return

    sequence = np.load(path).astype(np.float32)

    if sequence.ndim != 3:
        raise ValueError(
            f"Unexpected shape for {video_id}: "
            f"{sequence.shape}"
        )

    if sequence.shape[1] != 50:
        raise ValueError(
            f"Expected 50 landmarks for {video_id}, "
            f"got {sequence.shape[1]}"
        )

    num_frames = sequence.shape[0]

    frame_indices = select_frames(
        num_frames
    )

    print(
        f"{video_id}: "
        f"{num_frames} frames → "
        f"selected {frame_indices.tolist()}"
    )

    # --------------------------------------------------------
    # 2 rows × 5 columns
    #
    # Row 1 = left hand
    # Row 2 = right hand
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(16, 7)
    )

    fig.suptitle(
        f"Hand Skeleton Analysis — {video_id}\n"
        f"Actual: {case['actual']}   |   "
        f"Predicted: {case['predicted']}",
        fontsize=15
    )

    for column, frame_index in enumerate(frame_indices):

        frame = sequence[frame_index]

        left_hand = frame[
            LEFT_HAND_START:LEFT_HAND_END
        ]

        right_hand = frame[
            RIGHT_HAND_START:RIGHT_HAND_END
        ]

        # Normalize each hand independently ONLY
        # for visualization so finger configuration
        # is easier to compare.
        left_hand = normalize_hand(
            left_hand
        )

        right_hand = normalize_hand(
            right_hand
        )

        draw_hand(
            axes[0, column],
            left_hand,
            f"Left — Frame {frame_index}"
        )

        draw_hand(
            axes[1, column],
            right_hand,
            f"Right — Frame {frame_index}"
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{video_id}_hand_skeletons.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
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
    print("TCN HAND SKELETON ERROR VISUALIZATION")
    print("=" * 70)

    for case in ERROR_CASES:

        visualize_case(case)

        print()

    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()