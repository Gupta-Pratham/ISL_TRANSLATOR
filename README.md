# 🇮🇳 Indian Sign Language Recognition

### Efficient Spatio-Temporal Deep Learning for Isolated Indian Sign Language Recognition

A practical deep-learning system for recognizing isolated Indian Sign Language (ISL) gestures from video. The system uses **MediaPipe hand and upper-body pose landmarks** to represent each video as a temporal sequence, followed by a **Temporal Convolutional Network (TCN)** for 59-class word-level recognition.

The project covers the complete pipeline from video processing and landmark extraction to model training, evaluation, error analysis, and webcam inference.

---

## 📌 Overview

Recognizing sign language from video requires both **spatial information** about the hands and body and **temporal information** about how a gesture changes over time.

Instead of processing raw RGB video directly, this project extracts a compact landmark representation from each frame. The resulting sequence is then processed by a temporal deep-learning model.

```text
Input Video
     │
     ▼
MediaPipe Hand + Pose Detection
     │
     ▼
50 Landmarks / Frame
     │
     ▼
150 Features / Frame
     │
     ▼
Interpolation + Normalization
     │
     ▼
Temporal Convolutional Network
     │
     ▼
59-Class Classification
     │
     ▼
Predicted ISL Word
```

---

## 🎯 Objectives

- Build an end-to-end ISL recognition pipeline.
- Extract useful hand and upper-body information from video.
- Preserve temporal information across sign sequences.
- Train and compare different temporal deep-learning models.
- Evaluate the final model using standard classification metrics.
- Analyze common model errors and confusion patterns.
- Build a webcam-based demonstration application.
- Keep the system lightweight enough for practical inference.

---

## ✨ Key Features

- 🎥 Video-based ISL recognition
- ✋ Two-hand landmark extraction
- 🧍 Upper-body pose landmarks
- 🧹 Missing-landmark interpolation
- 📐 Shoulder-based spatial normalization
- 🧠 Temporal Convolutional Network
- 📊 Model comparison and evaluation
- 🔎 Error and confusion analysis
- 🎬 Raw-video inference
- 📷 Webcam inference
- ⚡ GPU-supported model inference
- 📦 Trained model checkpoint included in the repository

---

## 📚 Dataset

This project uses the **INCLUDE (Indian Sign Language Dataset)** developed by AI4Bharat.

The implementation uses the **Adjectives** portion of the dataset.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total videos | 791 |
| Classes | 59 |
| Total frames | 46,619 |
| Training videos | 553 |
| Validation videos | 119 |
| Test videos | 119 |
| Mean sequence length | 58.94 frames |
| Standard deviation | 9.81 |
| Minimum sequence length | 36 |
| Median sequence length | 58 |
| Maximum sequence length | 108 |

The train, validation, and test splits contain no overlapping videos.

### Evaluation Note

The metadata used for this implementation did not provide signer IDs. Therefore, the reported evaluation is **video-disjoint**, not a verified signer-independent evaluation.

The current system should therefore be described as an **isolated ISL word/sign recognition system** rather than a continuous sentence-level translator.

---

## 🧩 Landmark Representation

Each video frame is converted into a compact representation using MediaPipe.

### Hand Landmarks

Two hands are supported.

```text
21 landmarks × 3 coordinates per hand
```

Total:

```text
42 hand landmarks
```

### Upper-Body Pose

The following pose landmarks are used:

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

Total:

```text
8 pose landmarks
```

### Final Representation

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

A complete video is represented as a variable-length sequence:

```text
T × 150
```

where `T` is the number of frames.

---

## 🧹 Preprocessing

Raw landmark sequences may contain missing detections. The preprocessing pipeline handles these before model inference.

### Missing-value handling

- Internal gaps are filled using linear interpolation.
- Boundary gaps use the nearest valid observation.
- Coordinates with no valid observation throughout a sequence are filled with zero.

### Spatial normalization

The shoulder region is used as a reference.

The midpoint between the shoulders is treated as the origin, while shoulder distance is used for scale normalization.

```text
Shoulder Midpoint
       ↓
Translate landmarks
       ↓
Normalize by shoulder distance
```

### Validation

The preprocessing pipeline checks the generated sequences for invalid shapes, NaN values, and infinite values.

Final preprocessing result:

```text
Videos processed: 791 / 791
NaN values:        0
Infinite values:   0
Failed files:      0
```

---

## 🧠 Model Architecture

Four temporal models were implemented and evaluated using the same 150-dimensional landmark representation:

- Bidirectional LSTM
- Bidirectional GRU
- Temporal Convolutional Network
- Transformer

The final project uses the **Temporal Convolutional Network (TCN)**.

### TCN

```text
150-D Input Sequence
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
59 ISL Classes
```

### TCN Configuration

| Parameter | Value |
|---|---:|
| Input features | 150 |
| Hidden dimension | 128 |
| Temporal blocks | 4 |
| Dilations | 1, 2, 4, 8 |
| Dropout | 0.3 |
| Output classes | 59 |
| Parameters | 423,483 |

---

## 🏋️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Loss | Class-weighted Cross Entropy |
| LR scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 3 |
| Maximum epochs | 30 |
| Early stopping patience | 7 |
| Gradient clipping | 1.0 |
| Dropout | 0.3 |

Class weighting was used to account for differences in class frequency.

---

## 📊 Model Comparison

All four models were evaluated on the same held-out test set using the same 150-dimensional hand + pose representation.

| Model | Parameters | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| BiLSTM | 694,203 | 72.27% | 69.18% | 72.88% | 68.23% |
| BiGRU | 529,339 | 89.08% | 91.24% | 91.53% | 89.98% |
| **TCN** | **423,483** | **90.76%** | **90.14%** | **91.53%** | **90.13%** |
| Transformer | 292,155 | 85.71% | 85.14% | 86.44% | 83.67% |

The TCN is used as the final model in the application.

---

## 📈 Final Test Results

The final TCN was evaluated on 119 held-out test videos.

```text
Test samples:        119
Correct predictions: 108
Incorrect:            11
```

| Metric | Result |
|---|---:|
| Accuracy | **90.76%** |
| Macro Precision | **90.14%** |
| Macro Recall | **91.53%** |
| Macro F1 | **90.13%** |
| Weighted F1 | ~89.97% |

These results correspond to the **video-disjoint INCLUDE test split**.

---

## ⚡ Model Efficiency

The models were benchmarked on an NVIDIA RTX 3050 Laptop GPU using a representative 100-frame landmark sequence.

| Model | Parameters | Model Size | Mean Latency | Throughput |
|---|---:|---:|---:|---:|
| BiLSTM | 694,203 | 2.648 MB | 1.547 ms | 646.23 FPS |
| BiGRU | 529,339 | 2.019 MB | 0.814 ms | 1228.75 FPS |
| TCN | 423,483 | 1.623 MB | 2.695 ms | 371.07 FPS |
| Transformer | 292,155 | 1.364 MB | 1.755 ms | 569.87 FPS |

> **Benchmark note:** FPS represents model inference throughput on an already-extracted 100-frame landmark sequence. It is not end-to-end webcam FPS because video capture and MediaPipe processing are not included.

---

## 🔎 Error Analysis

The final TCN produced:

```text
108 correct predictions
11 incorrect predictions
```

Some observed confusion pairs were:

```text
narrow → wide
wide   → narrow
bad    → good
thin   → low
tall   → warm
cheap  → thick
fast   → bad
wet    → dry
dry    → loose
```

Several classes have very small numbers of test examples, so individual class-level results for low-support classes should be interpreted carefully.

Detailed evaluation reports and visualizations are stored in the `results/` directory.

---

## 🎥 Webcam Application

The repository includes a webcam-based demonstration application.

Run:

```powershell
python scripts\webcam_demo.py
```

### Controls

```text
S → Start recording
E → End recording and predict
R → Reset recording
Q → Quit
```

### Webcam Pipeline

```text
Webcam
   │
   ▼
OpenCV
   │
   ▼
MediaPipe Hand + Pose Detection
   │
   ▼
Landmark Sequence
   │
   ▼
Preprocessing
   │
   ▼
Trained TCN
   │
   ▼
Predicted ISL Word
   │
   ▼
Confidence Display
```

### Important

The 90.76% test accuracy is **not a webcam accuracy measurement**.

Live webcam input can differ from the INCLUDE test videos because of camera position, lighting, background, signer appearance, distance from the camera, signing style, and landmark detection quality.

---

## 🎬 Video Inference

### Preprocessed Landmark Inference

```powershell
python scripts\predict_video.py MVI_9531
```

### Raw Video Inference

```powershell
python scripts\predict_raw_video.py "data
aw\INCLUDE\Adjectives\99. healthy\MVI_9531.MOV"
```

The raw-video pipeline performs:

```text
Raw Video
   ↓
MediaPipe Landmark Extraction
   ↓
Interpolation
   ↓
Shoulder Normalization
   ↓
TCN
   ↓
Predicted ISL Word
```

---

## 🗂️ Repository Structure

```text
ISL_Translator/
│
├── data/
│   ├── raw/                 # Local dataset, not committed
│   ├── processed/           # Generated landmark data
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
│   ├── model_efficiency_benchmark.csv
│   └── ...
│
├── notebooks/
├── experiments/
├── training/
│
├── ARCHITECTURE.md
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Requirements

Recommended environment:

```text
Python 3.10
Windows / Linux
```

An NVIDIA GPU is optional for inference.

### 1. Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ISL_Translator
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scriptsctivate
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> PyTorch installation may vary depending on whether the system uses CPU or a compatible NVIDIA CUDA environment.

---

## 📦 Dataset Setup

The complete INCLUDE dataset is not stored in this repository because of its size.

The repository contains the runtime assets required for inference:

```text
models/
├── hand_landmarker.task
├── pose_landmarker_lite.task
└── checkpoints/
    └── tcn_baseline_best.pt
```

Therefore, teammates who only want to run the trained inference application do not need the complete training dataset.

The dataset is required to reproduce:

- Landmark extraction
- Preprocessing
- Model training
- Full evaluation

---

## ▶️ Run the Webcam Demo

After installing the dependencies:

```powershell
python scripts\webcam_demo.py
```

Make sure a working webcam is connected.

---

## 🧪 Train From Scratch

Training requires the INCLUDE dataset.

### Extract landmarks

```powershell
python scripts\extract_landmarks.py
```

### Preprocess landmarks

```powershell
python scripts\preprocess_landmarks.py
```

### Train the TCN

```powershell
python scripts	rain_tcn.py
```

The trained checkpoint is saved to:

```text
models/checkpoints/tcn_baseline_best.pt
```

---

## 📊 Evaluation

The repository includes scripts for:

- Test-set evaluation
- Classification metrics
- Confusion analysis
- Error analysis
- Model efficiency benchmarking

Generated reports and visualizations are stored under:

```text
results/
```

---

## ⚠️ Limitations

### 1. Video-disjoint evaluation

Signer IDs were not available in the metadata used for this implementation.

Therefore, the current evaluation does not establish signer-independent generalization.

### 2. Isolated signs

The final system recognizes isolated signs/words from 59 classes.

It does not perform continuous sentence-level ISL translation.

### 3. Limited vocabulary

The final classifier contains 59 classes from the selected INCLUDE Adjectives data.

### 4. Webcam domain shift

Performance on live webcam input can differ from the held-out dataset results.

### 5. Landmark dependence

The system depends on successful MediaPipe landmark detection. Occlusion, poor lighting, unusual camera angles, or missed detections can affect predictions.

---

## 🔮 Future Scope

Possible extensions include:

- Expanding the ISL vocabulary
- Continuous sign recognition
- Sentence-level translation
- Language-model integration
- Signer-independent evaluation with suitable metadata
- Larger ISL datasets
- Mobile deployment
- ONNX/TensorRT optimization
- Multilingual spoken-language output
- Improved handling of occlusion and missing landmarks

---

## 👥 Team Contribution

The project can be divided into three main workstreams.

### Data & Computer Vision

- Dataset preparation
- MediaPipe integration
- Landmark extraction
- Landmark preprocessing
- Visualization

### Deep Learning

- Dataset loader
- Sequence modelling
- TCN implementation
- Training pipeline
- Evaluation
- Model analysis

### Application & Integration

- Inference pipeline
- Webcam application
- Model integration
- Testing
- Documentation

All team members can contribute to final integration, testing, presentation, and documentation.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Landmark Detection | MediaPipe Tasks |
| Deep Learning | PyTorch |
| Sequence Model | Temporal Convolutional Network |
| Data Processing | NumPy, Pandas |
| ML Utilities | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Development | VS Code |
| Version Control | Git + GitHub |
| GPU Acceleration | NVIDIA CUDA |

---

## 📌 Project Workflow

```text
Dataset
   ↓
Video Processing
   ↓
Landmark Extraction
   ↓
Data Cleaning
   ↓
Spatial Normalization
   ↓
Sequence Modelling
   ↓
Model Training
   ↓
Evaluation
   ↓
Error Analysis
   ↓
Inference Application
```

---

## 📚 References

### INCLUDE Dataset

**AI4Bharat — INCLUDE**

https://github.com/AI4Bharat/INCLUDE

**Hugging Face Dataset**

https://huggingface.co/datasets/ai4bharat/INCLUDE

### Related ISL Research

**CISLR**

https://aclanthology.org/2022.emnlp-main.707/

**ISLTranslate**

https://aclanthology.org/2023.findings-acl.665/

**iSign**

https://aclanthology.org/2024.findings-acl.643/

### Frameworks

**MediaPipe**

https://ai.google.dev/edge/mediapipe

**PyTorch**

https://pytorch.org/

**OpenCV**

https://opencv.org/

---

## 🙏 Acknowledgements

This project uses and builds upon:

- AI4Bharat INCLUDE Dataset
- MediaPipe Tasks
- PyTorch
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib

---

## ⚖️ License

Add the appropriate license for the project before public redistribution.

The INCLUDE dataset and third-party model assets remain subject to their respective licenses and terms.

---

## ⭐ Final Implementation

The repository contains the completed project pipeline:

```text
✓ Landmark extraction
✓ Landmark preprocessing
✓ Dataset loading
✓ TCN model
✓ Training pipeline
✓ Evaluation
✓ Error analysis
✓ Efficiency benchmarking
✓ Raw-video inference
✓ Webcam inference
✓ Final trained checkpoint
```

Final trained model:

```text
models/checkpoints/tcn_baseline_best.pt
```

The trained inference application can be used without downloading the complete training dataset.
