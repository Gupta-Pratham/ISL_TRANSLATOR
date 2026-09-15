import sys
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import torch
import pandas as pd
import mediapipe as mp

from PIL import Image, ImageTk


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TCNClassifier


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "tcn_baseline_best.pt"
)

HAND_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)

POSE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pose_landmarker_lite.task"
)

NUM_CLASSES = 59
INPUT_SIZE = 150
HIDDEN_SIZE = 128

# Upper-body pose landmarks:
# left/right shoulder
# left/right elbow
# left/right wrist
# left/right hip
POSE_INDICES = [
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
]


# ============================================================
# LABEL MAP
# ============================================================

def load_labels():
    """
    Automatically find a CSV containing:
        class_id
        label

    This avoids depending on a hard-coded CSV path.
    """

    csv_files = list(
        PROJECT_ROOT.rglob("*.csv")
    )

    candidates = []

    for csv_path in csv_files:

        try:

            df = pd.read_csv(
                csv_path,
                nrows=5
            )

            required_columns = {
                "class_id",
                "label"
            }

            if required_columns.issubset(
                df.columns
            ):
                candidates.append(
                    csv_path
                )

        except Exception:
            continue

    if not candidates:

        raise FileNotFoundError(
            "Could not find a CSV containing "
            "'class_id' and 'label' columns."
        )

    # Prefer test.csv if available.
    test_candidates = [
        path
        for path in candidates
        if path.name.lower() == "test.csv"
    ]

    if test_candidates:
        csv_path = test_candidates[0]
    else:
        csv_path = candidates[0]

    df = pd.read_csv(
        csv_path
    )

    labels = (
        df[
            [
                "class_id",
                "label"
            ]
        ]
        .drop_duplicates()
        .sort_values("class_id")
    )

    label_map = dict(
        zip(
            labels["class_id"],
            labels["label"]
        )
    )

    if len(label_map) < NUM_CLASSES:

        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but found {len(label_map)} in:\n"
            f"{csv_path}"
        )

    print(
        f"Using label CSV:\n{csv_path}"
    )

    print(
        f"Loaded {len(label_map)} classes."
    )

    return label_map


# ============================================================
# LANDMARK PREPROCESSING
# ============================================================

def interpolate_landmarks(sequence):
    """
    Fill missing landmark coordinates.

    Input:
        (T, 50, 3)

    Output:
        (T, 50, 3)
    """

    sequence = sequence.astype(
        np.float32
    ).copy()

    T, num_landmarks, num_coordinates = (
        sequence.shape
    )

    for landmark_idx in range(
        num_landmarks
    ):

        for coordinate_idx in range(
            num_coordinates
        ):

            values = sequence[
                :,
                landmark_idx,
                coordinate_idx
            ]

            valid = np.isfinite(
                values
            )

            # No valid observation.
            if valid.sum() == 0:

                sequence[
                    :,
                    landmark_idx,
                    coordinate_idx
                ] = 0.0

                continue

            valid_indices = np.where(
                valid
            )[0]

            sequence[
                :,
                landmark_idx,
                coordinate_idx
            ] = np.interp(
                np.arange(T),
                valid_indices,
                values[valid]
            )

    return sequence


def normalize_landmarks(sequence):
    """
    Normalize using shoulder midpoint
    and shoulder distance.

    Shoulder indices:
        42 = left shoulder
        43 = right shoulder
    """

    sequence = sequence.astype(
        np.float32
    ).copy()

    left_shoulder = sequence[
        :,
        42,
        :
    ]

    right_shoulder = sequence[
        :,
        43,
        :
    ]

    # Shoulder midpoint.
    center = (
        left_shoulder
        + right_shoulder
    ) / 2.0

    # Translate origin to shoulder midpoint.
    sequence = (
        sequence
        - center[:, None, :]
    )

    # Shoulder distance.
    shoulder_distance = np.linalg.norm(
        left_shoulder
        - right_shoulder,
        axis=1
    )

    valid = (
        shoulder_distance
        > 1e-6
    )

    if np.any(valid):

        fallback_scale = np.median(
            shoulder_distance[valid]
        )

    else:

        fallback_scale = 1.0

    shoulder_distance[
        ~valid
    ] = fallback_scale

    # Scale normalization.
    sequence = (
        sequence
        / shoulder_distance[
            :,
            None,
            None
        ]
    )

    return sequence


def preprocess_landmarks(sequence):
    """
    Complete preprocessing pipeline:
        1. interpolation
        2. shoulder normalization
        3. NaN/Inf cleanup
    """

    sequence = interpolate_landmarks(
        sequence
    )

    sequence = normalize_landmarks(
        sequence
    )

    sequence = np.nan_to_num(
        sequence,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return sequence.astype(
        np.float32
    )


# ============================================================
# MEDIAPIPE
# ============================================================

def create_landmarkers():

    base_options = mp.tasks.BaseOptions

    hand_options = (
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options(
                model_asset_path=str(
                    HAND_MODEL_PATH
                )
            ),
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
            ),
            num_hands=2
        )
    )

    pose_options = (
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options(
                model_asset_path=str(
                    POSE_MODEL_PATH
                )
            ),
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
            ),
            num_poses=1
        )
    )

    hand_landmarker = (
        mp.tasks.vision.HandLandmarker
        .create_from_options(
            hand_options
        )
    )

    pose_landmarker = (
        mp.tasks.vision.PoseLandmarker
        .create_from_options(
            pose_options
        )
    )

    return (
        hand_landmarker,
        pose_landmarker
    )


def extract_frame_landmarks(
    frame,
    hand_landmarker,
    pose_landmarker,
    timestamp_ms
):
    """
    Extract:

        21 left-hand landmarks
        21 right-hand landmarks
        8 upper-body pose landmarks

    Total:

        50 landmarks
        150 coordinates
    """

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # --------------------------------------------------------
    # Hands
    # --------------------------------------------------------

    hand_result = (
        hand_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )

    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    pose_result = (
        pose_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )

    landmarks = np.full(
        (50, 3),
        np.nan,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Hand landmarks
    # --------------------------------------------------------

    if hand_result.hand_landmarks:

        for hand_idx, hand in enumerate(
            hand_result.hand_landmarks[:2]
        ):

            start_idx = (
                hand_idx * 21
            )

            for landmark_idx, landmark in enumerate(
                hand
            ):

                landmarks[
                    start_idx
                    + landmark_idx
                ] = [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]

    # --------------------------------------------------------
    # Pose landmarks
    # --------------------------------------------------------

    if pose_result.pose_landmarks:

        pose = (
            pose_result.pose_landmarks[0]
        )

        for i, pose_idx in enumerate(
            POSE_INDICES
        ):

            if pose_idx < len(pose):

                landmark = pose[
                    pose_idx
                ]

                landmarks[
                    42 + i
                ] = [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]

    return landmarks


# ============================================================
# MODEL
# ============================================================

def load_model(device):

    model = TCNClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        dropout=0.3
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    # Handle both:
    #   raw state_dict
    # and
    #   checkpoint dictionaries.
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_sequence(
    sequence,
    model,
    device,
    label_map
):
    """
    Predict one complete sign sequence.
    """

    # Same preprocessing used
    # during model preparation.
    sequence = preprocess_landmarks(
        sequence
    )

    # (T, 50, 3)
    # ->
    # (T, 150)
    features = sequence.reshape(
        sequence.shape[0],
        -1
    )

    x = torch.tensor(
        features,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    # No padding because this is
    # one complete recorded sequence.
    padding_mask = torch.zeros(
        (1, x.shape[1]),
        dtype=torch.bool,
        device=device
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

        confidence, prediction = (
            torch.max(
                probabilities,
                dim=1
            )
        )

    class_id = int(
        prediction.item()
    )

    confidence_value = float(
        confidence.item()
    )

    label = label_map.get(
        class_id,
        f"Class {class_id}"
    )

    return (
        label,
        confidence_value,
        class_id
    )


# ============================================================
# APPLICATION
# ============================================================

class ISLTranslatorApp:

    def __init__(self, root):

        self.root = root

        # ----------------------------------------------------
        # Safe initial state
        # ----------------------------------------------------

        self.cap = None

        self.hand_landmarker = None

        self.pose_landmarker = None

        self.recording = False

        self.landmark_buffer = []

        self.frame_count = 0

        # IMPORTANT:
        # MediaPipe VIDEO mode requires timestamps
        # to continuously increase.
        self.timestamp_ms = 0

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        self.root.title(
            "ISL Translator"
        )

        self.root.geometry(
            "1150x760"
        )

        self.root.minsize(
            950,
            650
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_app
        )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------------------
        # Load model and labels
        # ----------------------------------------------------

        try:

            self.label_map = (
                load_labels()
            )

            self.model = load_model(
                self.device
            )

        except Exception as e:

            messagebox.showerror(
                "Initialization Error",
                "Could not load the model or labels.\n\n"
                f"{e}"
            )

            self.close_app()

            return

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        try:

            (
                self.hand_landmarker,
                self.pose_landmarker
            ) = create_landmarkers()

        except Exception as e:

            messagebox.showerror(
                "MediaPipe Error",
                "Could not initialize MediaPipe.\n\n"
                f"{e}"
            )

            self.close_app()

            return

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        try:

            self.cap = (
                cv2.VideoCapture(0)
            )

            if not self.cap.isOpened():

                raise RuntimeError(
                    "Could not open webcam."
                )

        except Exception as e:

            messagebox.showerror(
                "Camera Error",
                str(e)
            )

            self.close_app()

            return

        # ----------------------------------------------------
        # Build UI
        # ----------------------------------------------------

        self.build_ui()

        # ----------------------------------------------------
        # Start camera loop
        # ----------------------------------------------------

        self.update_camera()

    # ========================================================
    # USER INTERFACE
    # ========================================================

    def build_ui(self):

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = ttk.Frame(
            self.root,
            padding=15
        )

        header.pack(
            fill="x"
        )

        title = ttk.Label(
            header,
            text="Indian Sign Language Translator",
            font=(
                "Segoe UI",
                24,
                "bold"
            )
        )

        title.pack()

        subtitle = ttk.Label(
            header,
            text=(
                "TCN-based ISL recognition "
                "using hand and upper-body landmarks"
            ),
            font=(
                "Segoe UI",
                10
            )
        )

        subtitle.pack(
            pady=(5, 0)
        )

        # ----------------------------------------------------
        # Main content
        # ----------------------------------------------------

        main = ttk.Frame(
            self.root,
            padding=15
        )

        main.pack(
            fill="both",
            expand=True
        )

        main.columnconfigure(
            0,
            weight=3
        )

        main.columnconfigure(
            1,
            weight=2
        )

        main.rowconfigure(
            0,
            weight=1
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        camera_frame = ttk.LabelFrame(
            main,
            text="Live Camera",
            padding=10
        )

        camera_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        camera_frame.rowconfigure(
            0,
            weight=1
        )

        camera_frame.columnconfigure(
            0,
            weight=1
        )

        self.camera_label = ttk.Label(
            camera_frame,
            text="Starting camera..."
        )

        self.camera_label.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        # ----------------------------------------------------
        # Recognition panel
        # ----------------------------------------------------

        result_frame = ttk.LabelFrame(
            main,
            text="Recognition Result",
            padding=20
        )

        result_frame.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        result_frame.columnconfigure(
            0,
            weight=1
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        ttk.Label(
            result_frame,
            text="STATUS",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).grid(
            row=0,
            column=0,
            pady=(10, 5)
        )

        self.status_label = ttk.Label(
            result_frame,
            text="Ready",
            font=(
                "Segoe UI",
                15
            )
        )

        self.status_label.grid(
            row=1,
            column=0,
            pady=(0, 25)
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        ttk.Label(
            result_frame,
            text="RECOGNIZED SIGN",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).grid(
            row=2,
            column=0,
            pady=(10, 5)
        )

        self.prediction_label = ttk.Label(
            result_frame,
            text="—",
            font=(
                "Segoe UI",
                22,
                "bold"
            ),
            anchor="center",
            wraplength=330
        )

        self.prediction_label.grid(
            row=3,
            column=0,
            pady=(0, 25)
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        ttk.Label(
            result_frame,
            text="CONFIDENCE",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).grid(
            row=4,
            column=0,
            pady=(5, 5)
        )

        self.confidence_label = ttk.Label(
            result_frame,
            text="—",
            font=(
                "Segoe UI",
                18
            )
        )

        self.confidence_label.grid(
            row=5,
            column=0,
            pady=(0, 25)
        )

        # ----------------------------------------------------
        # Frames
        # ----------------------------------------------------

        self.frame_label = ttk.Label(
            result_frame,
            text="Frames: 0",
            font=(
                "Segoe UI",
                10
            )
        )

        self.frame_label.grid(
            row=6,
            column=0,
            pady=5
        )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        device_name = (
            "NVIDIA CUDA"
            if self.device.type == "cuda"
            else "CPU"
        )

        self.device_label = ttk.Label(
            result_frame,
            text=f"Model device: {device_name}",
            font=(
                "Segoe UI",
                9
            )
        )

        self.device_label.grid(
            row=7,
            column=0,
            pady=5
        )

        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        controls = ttk.Frame(
            self.root,
            padding=(15, 5)
        )

        controls.pack(
            fill="x"
        )

        self.start_button = ttk.Button(
            controls,
            text="▶  Start Sign",
            command=self.start_recording
        )

        self.start_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=5
        )

        self.end_button = ttk.Button(
            controls,
            text="■  End & Predict",
            command=self.end_recording,
            state="disabled"
        )

        self.end_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=5
        )

        self.reset_button = ttk.Button(
            controls,
            text="↻  Reset",
            command=self.reset_recording
        )

        self.reset_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=5
        )

        self.quit_button = ttk.Button(
            controls,
            text="✕  Quit",
            command=self.close_app
        )

        self.quit_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=5
        )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        instructions = ttk.Label(
            self.root,
            text=(
                "Click Start Sign → perform one sign → "
                "click End & Predict"
            ),
            anchor="center",
            padding=10
        )

        instructions.pack(
            fill="x"
        )

    # ========================================================
    # CAMERA LOOP
    # ========================================================

    def update_camera(self):

        if self.cap is None:

            return

        if not self.cap.isOpened():

            return

        ret, frame = (
            self.cap.read()
        )

        if not ret:

            self.root.after(
                30,
                self.update_camera
            )

            return

        # Mirror camera.
        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # Process recording
        # ----------------------------------------------------

        if self.recording:

            # Use a continuously increasing timestamp.
            # Do NOT reset this when a new sign starts.
            self.timestamp_ms += 33

            landmarks = (
                extract_frame_landmarks(
                    frame,
                    self.hand_landmarker,
                    self.pose_landmarker,
                    self.timestamp_ms
                )
            )

            self.landmark_buffer.append(
                landmarks
            )

            self.frame_count += 1

            self.frame_label.config(
                text=(
                    f"Frames: "
                    f"{len(self.landmark_buffer)}"
                )
            )

            # Recording indicator.
            cv2.circle(
                frame,
                (30, 30),
                10,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "RECORDING",
                (50, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # Convert OpenCV image to Tkinter
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        # Keep camera display manageable.
        image.thumbnail(
            (700, 520)
        )

        photo = ImageTk.PhotoImage(
            image=image
        )

        self.camera_label.configure(
            image=photo
        )

        self.camera_label.image = photo

        # Continue loop.
        self.root.after(
            15,
            self.update_camera
        )

    # ========================================================
    # START RECORDING
    # ========================================================

    def start_recording(self):

        self.landmark_buffer = []

        self.frame_count = 0

        self.recording = True

        self.status_label.config(
            text="Recording..."
        )

        self.prediction_label.config(
            text="—"
        )

        self.confidence_label.config(
            text="—"
        )

        self.frame_label.config(
            text="Frames: 0"
        )

        self.start_button.config(
            state="disabled"
        )

        self.end_button.config(
            state="normal"
        )

    # ========================================================
    # END RECORDING + PREDICT
    # ========================================================

    def end_recording(self):

        self.recording = False

        self.start_button.config(
            state="normal"
        )

        self.end_button.config(
            state="disabled"
        )

        number_of_frames = len(
            self.landmark_buffer
        )

        # Minimum safety check.
        if number_of_frames < 10:

            self.status_label.config(
                text="Too few frames"
            )

            messagebox.showwarning(
                "Recording Too Short",
                "Please perform the sign for a little longer."
            )

            return

        self.status_label.config(
            text="Predicting..."
        )

        self.root.update_idletasks()

        try:

            sequence = np.stack(
                self.landmark_buffer,
                axis=0
            )

            label, confidence, class_id = (
                predict_sequence(
                    sequence,
                    self.model,
                    self.device,
                    self.label_map
                )
            )

            self.prediction_label.config(
                text=label
            )

            self.confidence_label.config(
                text=(
                    f"{confidence * 100:.2f}%"
                )
            )

            self.status_label.config(
                text="Prediction complete"
            )

            print(
                "\nPrediction"
            )

            print(
                f"Class ID: {class_id}"
            )

            print(
                f"Label: {label}"
            )

            print(
                f"Confidence: "
                f"{confidence:.4f}"
            )

        except Exception as e:

            self.status_label.config(
                text="Prediction error"
            )

            messagebox.showerror(
                "Prediction Error",
                str(e)
            )

    # ========================================================
    # RESET
    # ========================================================

    def reset_recording(self):

        self.recording = False

        self.landmark_buffer = []

        self.frame_count = 0

        self.status_label.config(
            text="Ready"
        )

        self.prediction_label.config(
            text="—"
        )

        self.confidence_label.config(
            text="—"
        )

        self.frame_label.config(
            text="Frames: 0"
        )

        self.start_button.config(
            state="normal"
        )

        self.end_button.config(
            state="disabled"
        )

    # ========================================================
    # CLOSE APPLICATION
    # ========================================================

    def close_app(self):

        self.recording = False

        # Camera.
        if getattr(
            self,
            "cap",
            None
        ) is not None:

            try:
                self.cap.release()
            except Exception:
                pass

        # MediaPipe.
        if getattr(
            self,
            "hand_landmarker",
            None
        ) is not None:

            try:
                self.hand_landmarker.close()
            except Exception:
                pass

        if getattr(
            self,
            "pose_landmarker",
            None
        ) is not None:

            try:
                self.pose_landmarker.close()
            except Exception:
                pass

        try:
            self.root.destroy()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    try:

        style = ttk.Style()

        style.theme_use(
            "clam"
        )

    except tk.TclError:
        pass

    ISLTranslatorApp(
        root
    )

    root.mainloop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()