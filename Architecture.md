# System Architecture

## 1. Overview

The system is an isolated Indian Sign Language recognition pipeline that converts video into a sequence of hand and upper-body landmarks and classifies the resulting temporal sequence using a Temporal Convolutional Network (TCN).

```text
Input Video
    ↓
MediaPipe Hand + Pose Detection
    ↓
Landmark Sequence
    ↓
Missing-Value Interpolation
    ↓
Shoulder-Based Normalization
    ↓
150-Dimensional Temporal Sequence
    ↓
Temporal Convolutional Network
    ↓
59-Class Classification
    ↓
Predicted ISL Word
```

---

## 2. End-to-End Architecture

```text
┌───────────────────────────────┐
│           Input Video         │
│     INCLUDE / Webcam Input    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       MediaPipe Tasks         │
│                               │
│   Hand Landmarker             │
│   Pose Landmarker Lite        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│    Landmark Representation    │
│                               │
│  42 hand landmarks            │
│  8 upper-body pose landmarks  │
│                               │
│  50 landmarks × 3 coordinates │
│  = 150 features / frame       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│         Preprocessing         │
│                               │
│  • Missing-value handling     │
│  • Temporal interpolation     │
│  • Shoulder normalization     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│          TCN Model            │
│                               │
│  Input Projection             │
│  Dilated Temporal Blocks      │
│  Residual Connections         │
│  Masked Mean Pooling          │
│  Classification Head          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       59-Class Output         │
│                               │
│       Predicted ISL Word      │
│       + Confidence            │
└───────────────────────────────┘
```

---

## 3. Input Layer

The system accepts video sequences.

For the dataset pipeline, videos are processed frame by frame.

For the live application, frames are captured continuously from a webcam using OpenCV.

The same landmark representation and preprocessing logic are used for both offline and webcam inference.

---

## 4. Landmark Extraction

MediaPipe Tasks is used to extract hand and upper-body pose information.

### 4.1 Hand Landmarks

The system supports two hands.

Each hand provides:

```text
21 landmarks × 3 coordinates
```

Therefore:

```text
42 hand landmarks
```

### 4.2 Pose Landmarks

The selected upper-body pose landmarks are:

```text
11 - Left Shoulder
12 - Right Shoulder
13 - Left Elbow
14 - Right Elbow
15 - Left Wrist
16 - Right Wrist
23 - Left Hip
24 - Right Hip
```

Therefore:

```text
8 pose landmarks
```

### 4.3 Combined Representation

```text
42 hand landmarks
+
8 pose landmarks
-----------------
50 landmarks / frame
```

Each landmark contains:

```text
x, y, z
```

Therefore:

```text
50 × 3 = 150 features / frame
```

A video containing `T` frames becomes:

```text
T × 150
```

where `T` varies between videos.

---

## 5. Preprocessing

The raw landmark representation can contain missing detections because MediaPipe may not detect every landmark in every frame.

The preprocessing stage handles these missing values before model training or inference.

### 5.1 Temporal Interpolation

For each coordinate independently:

- Internal missing values are linearly interpolated.
- Missing values at the beginning or end use the nearest valid observation.
- Coordinates with no valid observation across the entire sequence are filled with zero.

Conceptually:

```text
Valid ── Missing ── Missing ── Valid
              ↓
       Linear interpolation
```

### 5.2 Shoulder-Based Normalization

The shoulders provide a body-relative reference.

Let:

```text
S_left  = left shoulder
S_right = right shoulder
```

The shoulder midpoint is:

```text
C = (S_left + S_right) / 2
```

The landmark coordinates are translated relative to this midpoint.

The distance between the shoulders is used as the scale:

```text
D = ||S_left - S_right||
```

The normalized representation follows the form:

```text
P_normalized = (P - C) / D
```

This reduces sensitivity to the absolute position and scale of the signer within the camera view.

### 5.3 Validation

Processed sequences are checked for:

- Valid dimensions
- NaN values
- Infinite values

The final preprocessing run successfully processed:

```text
791 / 791 videos
0 NaN values
0 infinite values
0 failed files
```

---

## 6. Sequence Representation

After preprocessing, every video is represented as a variable-length sequence:

```text
(T, 50, 3)
```

The PyTorch dataset converts this into:

```text
(T, 150)
```

because:

```text
50 landmarks × 3 coordinates = 150 features
```

Different videos can contain different numbers of frames.

During batching, sequences are padded and a padding mask is created so that padded frames are not treated as real observations during pooling.

---

## 7. TCN Architecture

The final classifier is a Temporal Convolutional Network.

```text
Input
T × 150
   │
   ▼
Input Projection
   │
   ▼
Temporal Block
Dilation = 1
   │
   ▼
Temporal Block
Dilation = 2
   │
   ▼
Temporal Block
Dilation = 4
   │
   ▼
Temporal Block
Dilation = 8
   │
   ▼
Masked Mean Pooling
   │
   ▼
Classification Head
   │
   ▼
59 Class Logits
```

### Model Configuration

| Parameter | Value |
|---|---:|
| Input features | 150 |
| Hidden dimension | 128 |
| Temporal blocks | 4 |
| Dilations | 1, 2, 4, 8 |
| Dropout | 0.3 |
| Output classes | 59 |
| Parameters | 423,483 |

### Temporal Blocks

The temporal blocks use increasing dilation factors:

```text
1 → 2 → 4 → 8
```

This allows the model to process temporal information over different receptive-field sizes while keeping the architecture relatively compact.

Residual connections are used within the temporal blocks.

---

## 8. Masked Mean Pooling

Because videos have different numbers of frames, batches are padded to a common sequence length.

The dataset loader creates a padding mask.

The model uses masked mean pooling so that padded frames do not contribute to the final sequence representation.

Conceptually:

```text
Variable-Length Sequences
          ↓
       Padding
          ↓
     Padding Mask
          ↓
      TCN Features
          ↓
    Masked Mean Pooling
          ↓
 Sequence Representation
```

---

## 9. Classification Head

The pooled temporal representation is passed through the classification head.

The output dimension is:

```text
59
```

Each output corresponds to one of the 59 ISL classes used in the experiment.

The predicted class is obtained from the highest-scoring output.

The inference application also reports the associated prediction confidence.

---

## 10. Training Pipeline

```text
Training CSV
     │
     ▼
PyTorch Dataset
     │
     ▼
Preprocessed .npy Sequence
     │
     ▼
Batch Padding + Mask
     │
     ▼
TCN
     │
     ▼
Class-Weighted Cross Entropy
     │
     ▼
AdamW Optimizer
     │
     ▼
Validation
     │
     ├───────────────┐
     ▼               ▼
LR Scheduler     Best Model
                     │
                     ▼
              Early Stopping
```

### Training Configuration

```text
Optimizer: AdamW
Learning rate: 1e-3
Weight decay: 1e-4
Loss: Class-weighted Cross Entropy
LR scheduler: ReduceLROnPlateau
Scheduler factor: 0.5
Scheduler patience: 3
Maximum epochs: 30
Early stopping patience: 7
Gradient clipping: 1.0
Dropout: 0.3
```

---

## 11. Evaluation Pipeline

The final evaluation uses the held-out test videos.

```text
Test CSV
   │
   ▼
Test Landmark Sequence
   │
   ▼
Preprocessing
   │
   ▼
Trained TCN
   │
   ▼
Predicted Class
   │
   ▼
Compare with Ground Truth
   │
   ▼
Accuracy / Precision / Recall / F1
```

The reported test split contains:

```text
119 test videos
```

with:

```text
108 correct predictions
11 incorrect predictions
```

Final reported metrics:

```text
Accuracy:        90.76%
Macro Precision: 90.14%
Macro Recall:    91.53%
Macro F1:        90.13%
```

---

## 12. Model Comparison

Four temporal architectures were evaluated:

```text
BiLSTM
BiGRU
TCN
Transformer
```

All models use the same 150-dimensional hand + pose representation.

| Model | Parameters | Accuracy | Macro F1 |
|---|---:|---:|---:|
| BiLSTM | 694,203 | 72.27% | 68.23% |
| BiGRU | 529,339 | 89.08% | 89.98% |
| TCN | 423,483 | 90.76% | 90.13% |
| Transformer | 292,155 | 85.71% | 83.67% |

The TCN checkpoint is used by the final inference applications.

---

## 13. Offline Inference

The project supports inference from preprocessed landmark files.

```text
.npy Sequence
     │
     ▼
Dataset / Loader
     │
     ▼
TCN Checkpoint
     │
     ▼
Predicted Class
     │
     ▼
ISL Word
```

The repository provides:

```text
scripts/predict_video.py
```

---

## 14. Raw Video Inference

Raw videos can also be passed through the complete pipeline.

```text
Raw Video
    │
    ▼
MediaPipe
    │
    ▼
Raw Landmarks
    │
    ▼
Interpolation
    │
    ▼
Shoulder Normalization
    │
    ▼
TCN
    │
    ▼
Predicted ISL Word
```

Implementation:

```text
scripts/predict_raw_video.py
```

---

## 15. Webcam Inference

The webcam application uses the same landmark representation and preprocessing steps.

```text
Webcam
   │
   ▼
OpenCV Capture
   │
   ▼
MediaPipe Hand + Pose
   │
   ▼
Frame Landmark Buffer
   │
   ▼
Preprocessing
   │
   ▼
TCN
   │
   ▼
Prediction
   │
   ▼
Word + Confidence
```

Implementation:

```text
scripts/webcam_demo.py
```

### Controls

```text
S → Start recording
E → End recording and predict
R → Reset
Q → Quit
```

---

## 16. Runtime Model Assets

The final runtime requires:

```text
models/
├── hand_landmarker.task
├── pose_landmarker_lite.task
└── checkpoints/
    └── tcn_baseline_best.pt
```

The trained classifier checkpoint is:

```text
tcn_baseline_best.pt
```

The MediaPipe task files are used for landmark extraction.

---

## 17. Project Directory Architecture

```text
ISL_Translator/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── models/
│   ├── checkpoints/
│   │   └── tcn_baseline_best.pt
│   ├── hand_landmarker.task
│   ├── pose_landmarker_lite.task
│   └── README.md
│
├── scripts/
│   ├── extract_landmarks.py
│   ├── preprocess_landmarks.py
│   ├── train_tcn.py
│   ├── predict_video.py
│   ├── predict_raw_video.py
│   ├── webcam_demo.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── ...
│
├── src/
│   ├── dataset.py
│   └── models.py
│
├── results/
│   ├── reports/
│   ├── visualizations/
│   └── model_efficiency_benchmark.csv
│
├── notebooks/
├── experiments/
├── training/
│
├── README.md
├── ARCHITECTURE.md
└── requirements.txt
```

---

## 18. Evaluation Boundary

The current experiment evaluates:

```text
INCLUDE Videos
      ↓
Video-Level Split
      ↓
Landmark Extraction
      ↓
Preprocessing
      ↓
TCN Classification
      ↓
59-Class Test Evaluation
```

The available metadata did not provide signer IDs, so the current results establish **video-disjoint evaluation**, not verified signer-independent generalization.

The final application is therefore best described as an **isolated ISL word/sign recognition system**.

---

## 19. Main Limitations

### Dataset

The complete INCLUDE dataset is not included in the repository because of its size.

### Signer Independence

Signer-independent generalization was not directly evaluated because signer IDs were unavailable in the metadata used for the experiment.

### Vocabulary

The final classifier covers 59 classes from the selected INCLUDE Adjectives data.

### Continuous Translation

The system recognizes isolated signs/words. It does not currently perform continuous sentence-level ISL translation.

### Webcam Generalization

The offline test results should not be treated as a measurement of real-world webcam accuracy.

---

## 20. Summary

The final architecture combines:

```text
Computer Vision
       +
Landmark-Based Representation
       +
Temporal Deep Learning
       +
Sequence Classification
       +
Real-Time Inference
```

The resulting pipeline provides a compact and practical approach to isolated Indian Sign Language recognition while keeping the complete workflow reproducible and suitable for further development.
