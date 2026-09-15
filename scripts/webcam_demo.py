import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

HAND_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
POSE_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "tcn_baseline_best.pt"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)


# ============================================================
# SETTINGS
# ============================================================

POSE_INDICES = [
    11, 12,
    13, 14,
    15, 16,
    23, 24
]

INPUT_SIZE = 150
NUM_CLASSES = 59

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LABEL MAP
# ============================================================

import pandas as pd

test_df = pd.read_csv(TEST_CSV)

label_map = (
    test_df[["class_id", "label"]]
    .drop_duplicates()
    .set_index("class_id")["label"]
    .to_dict()
)


# ============================================================
# MEDIAPIPE
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

    return vision.HandLandmarker.create_from_options(
        options
    )


def create_pose_detector():

    base_options = python.BaseOptions(
        model_asset_path=str(POSE_MODEL)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    return vision.PoseLandmarker.create_from_options(
        options
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_landmarks(sequence):

    sequence = sequence.astype(np.float32)

    T, L, C = sequence.shape

    # --------------------------------------------------------
    # Interpolation
    # --------------------------------------------------------

    processed = sequence.copy()

    for landmark_idx in range(L):

        for coordinate_idx in range(C):

            values = processed[
                :,
                landmark_idx,
                coordinate_idx
            ]

            valid = ~np.isnan(values)

            if valid.any():

                valid_indices = np.where(valid)[0]

                processed[
                    :,
                    landmark_idx,
                    coordinate_idx
                ] = np.interp(
                    np.arange(T),
                    valid_indices,
                    values[valid]
                )

            else:

                processed[
                    :,
                    landmark_idx,
                    coordinate_idx
                ] = 0.0

    # --------------------------------------------------------
    # Shoulder normalization
    # --------------------------------------------------------

    LEFT_SHOULDER = 42
    RIGHT_SHOULDER = 43

    left_shoulder = processed[
        :,
        LEFT_SHOULDER,
        :
    ]

    right_shoulder = processed[
        :,
        RIGHT_SHOULDER,
        :
    ]

    shoulder_midpoint = (
        left_shoulder
        + right_shoulder
    ) / 2.0

    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder,
        axis=1
    )

    shoulder_distance[
        shoulder_distance < 1e-6
    ] = 1.0

    processed = (
        processed
        - shoulder_midpoint[:, None, :]
    )

    processed = (
        processed
        / shoulder_distance[:, None, None]
    )

    return processed


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    from src.models import TCNClassifier

    model = TCNClassifier(
        input_size=INPUT_SIZE,
        hidden_size=128,
        num_classes=NUM_CLASSES,
        dropout=0.3
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(sequence, model):

    sequence = sequence.reshape(
        sequence.shape[0],
        -1
    )

    x = torch.from_numpy(
        sequence
    ).unsqueeze(0).to(DEVICE)

    padding_mask = torch.zeros(
        1,
        sequence.shape[0],
        dtype=torch.bool,
        device=DEVICE
    )

    with torch.no_grad():

        logits = model(
            x,
            padding_mask
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    predicted_label = label_map.get(
        predicted_class,
        f"class_{predicted_class}"
    )

    return predicted_label, confidence


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ISL REAL-TIME WEBCAM DEMO")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print()
    print("Controls:")
    print("  S = Start recording")
    print("  E = End recording + predict")
    print("  R = Reset")
    print("  Q = Quit")
    print("=" * 60)

    model = load_model()

    hand_detector = create_hand_detector()
    pose_detector = create_pose_detector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    recording = False
    frames = []

    predicted_label = ""
    confidence = 0.0

    frame_index = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            display = frame.copy()

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            timestamp_ms = int(
                (frame_index / fps) * 1000
            )

            # ------------------------------------------------
            # HANDS
            # ------------------------------------------------

            hand_result = (
                hand_detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            left_hand = np.full(
                (21, 3),
                np.nan,
                dtype=np.float32
            )

            right_hand = np.full(
                (21, 3),
                np.nan,
                dtype=np.float32
            )

            if hand_result.hand_landmarks:

                for hand_idx, landmarks in enumerate(
                    hand_result.hand_landmarks
                ):

                    handedness = (
                        hand_result
                        .handedness[hand_idx][0]
                        .category_name
                    )

                    points = np.array(
                        [
                            [lm.x, lm.y, lm.z]
                            for lm in landmarks
                        ],
                        dtype=np.float32
                    )

                    if handedness.lower() == "left":

                        left_hand = points

                    elif handedness.lower() == "right":

                        right_hand = points

            # ------------------------------------------------
            # POSE
            # ------------------------------------------------

            pose_points = np.full(
                (len(POSE_INDICES), 3),
                np.nan,
                dtype=np.float32
            )

            pose_result = (
                pose_detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            if pose_result.pose_landmarks:

                landmarks = pose_result.pose_landmarks[0]

                for i, landmark_idx in enumerate(
                    POSE_INDICES
                ):

                    lm = landmarks[landmark_idx]

                    pose_points[i] = [
                        lm.x,
                        lm.y,
                        lm.z
                    ]

            # ------------------------------------------------
            # COMBINE
            # ------------------------------------------------

            combined = np.concatenate(
                [
                    left_hand,
                    right_hand,
                    pose_points
                ],
                axis=0
            )

            if recording:

                frames.append(combined)

            # ------------------------------------------------
            # UI
            # ------------------------------------------------

            if recording:

                cv2.putText(
                    display,
                    "RECORDING...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"Frames: {len(frames)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    display,
                    "Press S to record a sign",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            if predicted_label:

                cv2.putText(
                    display,
                    f"Prediction: {predicted_label}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"Confidence: {confidence:.2%}",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(
                "ISL Translator",
                display
            )

            # ------------------------------------------------
            # KEYBOARD
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            elif key == ord("s"):

                frames = []

                predicted_label = ""

                confidence = 0.0

                recording = True

                print("\nRecording started...")

            elif key == ord("e"):

                if recording and len(frames) > 0:

                    recording = False

                    print(
                        f"Recording ended: "
                        f"{len(frames)} frames"
                    )

                    sequence = np.stack(frames)

                    processed = preprocess_landmarks(
                        sequence
                    )

                    predicted_label, confidence = predict(
                        processed,
                        model
                    )

                    print(
                        f"Prediction: "
                        f"{predicted_label}"
                    )

                    print(
                        f"Confidence: "
                        f"{confidence:.4f}"
                    )

            elif key == ord("r"):

                frames = []

                recording = False

                predicted_label = ""

                confidence = 0.0

                print("\nReset.")

            frame_index += 1

    finally:

        cap.release()

        hand_detector.close()
        pose_detector.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()