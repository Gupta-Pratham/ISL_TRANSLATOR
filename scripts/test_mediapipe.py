import cv2
import mediapipe as mp
import os
import glob

# --------------------------------------------------
# Paths
# --------------------------------------------------

VIDEO = glob.glob(
    r".\data\raw\INCLUDE\Adjectives\1. loud\*.MOV"
)[0]

HAND_MODEL = r".\models\hand_landmarker.task"
POSE_MODEL = r".\models\pose_landmarker_lite.task"

OUTPUT = r".\results\mediapipe_test.mp4"


# --------------------------------------------------
# MediaPipe Tasks
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    fps,
    (width, height),
)


# --------------------------------------------------
# Draw functions
# --------------------------------------------------

def draw_landmarks(frame, landmarks, connections, color):

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                frame,
                (x, y),
                4,
                color,
                -1,
            )

    for connection in connections:

        start_idx, end_idx = connection

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * width)
        y1 = int(start.y * height)

        x2 = int(end.x * width)
        y2 = int(end.y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )


# --------------------------------------------------
# Landmark connections
# --------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]


# --------------------------------------------------
# Process video
# --------------------------------------------------

frame_number = 0

with HandLandmarker.create_from_options(hand_options) as hand_detector, \
     PoseLandmarker.create_from_options(pose_options) as pose_detector:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = int(
            frame_number * 1000 / fps
        )

        hand_result = hand_detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        pose_result = pose_detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        # ------------------------------------------
        # Draw hands
        # ------------------------------------------

        if hand_result.hand_landmarks:

            for hand_landmarks in hand_result.hand_landmarks:

                draw_landmarks(
                    frame,
                    hand_landmarks,
                    HAND_CONNECTIONS,
                    (0, 255, 0),
                )

        # ------------------------------------------
        # Draw pose
        # ------------------------------------------

        if pose_result.pose_landmarks:

            pose_landmarks = pose_result.pose_landmarks[0]

            draw_landmarks(
                frame,
                pose_landmarks,
                POSE_CONNECTIONS,
                (255, 0, 0),
            )

        # ------------------------------------------
        # Frame information
        # ------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        frame_number += 1


cap.release()
writer.release()

print()
print("===================================")
print(" MediaPipe Visualization Complete")
print("===================================")
print("Input :", os.path.basename(VIDEO))
print("Output:", OUTPUT)
print("Frames:", frame_number)
print("===================================")