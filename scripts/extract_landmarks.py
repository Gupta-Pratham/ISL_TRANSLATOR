import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path(r".\data\raw\INCLUDE\Adjectives")
OUTPUT_DIR = Path(r".\data\processed\landmarks_clean")

HAND_MODEL = Path(r".\models\hand_landmarker.task")
POSE_MODEL = Path(r".\models\pose_landmarker_lite.task")


# ============================================================
# SETTINGS
# ============================================================

# We only keep upper-body pose landmarks.
# MediaPipe Pose landmark indices:
#
# 11 = left shoulder
# 12 = right shoulder
# 13 = left elbow
# 14 = right elbow
# 15 = left wrist
# 16 = right wrist
# 23 = left hip
# 24 = right hip

POSE_INDICES = [
    11, 12,
    13, 14,
    15, 16,
    23, 24
]

NUM_TEST_VIDEOS = None


# ============================================================
# MEDIAPIPE
# ============================================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# HAND DETECTOR
# ============================================================

def create_hand_detector():

    base_options = python.BaseOptions(
        model_asset_path=str(HAND_MODEL)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2
    )

    return vision.HandLandmarker.create_from_options(options)


# ============================================================
# POSE DETECTOR
# ============================================================

def create_pose_detector():

    base_options = python.BaseOptions(
        model_asset_path=str(POSE_MODEL)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    return vision.PoseLandmarker.create_from_options(options)


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    print(f"\nProcessing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    frames = []

    # Fresh detectors for every video.
    # This avoids MediaPipe timestamp conflicts.
    hand_detector = create_hand_detector()
    pose_detector = create_pose_detector()

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # OpenCV: BGR
        # MediaPipe: RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int((frame_index / fps) * 1000)

        # ----------------------------------------------------
        # HANDS
        # ----------------------------------------------------

        hand_result = hand_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        left_hand = np.full((21, 3), np.nan, dtype=np.float32)
        right_hand = np.full((21, 3), np.nan, dtype=np.float32)

        if hand_result.hand_landmarks:

            for hand_idx, landmarks in enumerate(
                hand_result.hand_landmarks
            ):

                # Determine handedness
                handedness = hand_result.handedness[hand_idx][0].category_name

                points = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks],
                    dtype=np.float32
                )

                if handedness.lower() == "left":
                    left_hand = points

                elif handedness.lower() == "right":
                    right_hand = points

        # ----------------------------------------------------
        # POSE
        # ----------------------------------------------------

        pose_points = np.full(
            (len(POSE_INDICES), 3),
            np.nan,
            dtype=np.float32
        )

        if pose_result := pose_detector.detect_for_video(
            mp_image,
            timestamp_ms
        ):

            if pose_result.pose_landmarks:

                landmarks = pose_result.pose_landmarks[0]

                for i, landmark_idx in enumerate(POSE_INDICES):

                    lm = landmarks[landmark_idx]

                    pose_points[i] = [
                        lm.x,
                        lm.y,
                        lm.z
                    ]

        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        combined = np.concatenate(
            [
                left_hand,
                right_hand,
                pose_points
            ],
            axis=0
        )

        frames.append(combined)

        frame_index += 1

    cap.release()

    hand_detector.close()
    pose_detector.close()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    if not frames:
        print("ERROR: No frames extracted.")
        return

    sequence = np.stack(frames)

    relative_path = video_path.relative_to(INPUT_DIR)

    output_path = OUTPUT_DIR / relative_path
    output_path = output_path.with_suffix(".npy")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(output_path, sequence)

    print(
        f"Saved: {output_path.name} | "
        f"shape={sequence.shape} | "
        f"NaNs={np.isnan(sequence).sum()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    video_files = sorted(
        INPUT_DIR.rglob("*.MOV")
    )

    if not video_files:
        print("No .MOV files found.")
        return

    if NUM_TEST_VIDEOS is not None:
        video_files = video_files[:NUM_TEST_VIDEOS]

    print("============================================")
    print("ISL Landmark Extraction - Clean Representation")
    print("============================================")
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Videos: {len(video_files)}")
    print("Landmarks/frame: 50")
    print("Features/frame : 150")
    print("============================================")

    for video_path in video_files:
        process_video(video_path)

    print("\nExtraction test complete.")


if __name__ == "__main__":
    main()