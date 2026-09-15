# Models

This directory contains the runtime assets used by the ISL recognition application.

## MediaPipe Assets

```text
hand_landmarker.task
pose_landmarker_lite.task
```

These files are used by the MediaPipe Tasks pipeline for hand and upper-body pose landmark extraction.

## Final Classifier

```text
checkpoints/
└── tcn_baseline_best.pt
```

The final classifier is a Temporal Convolutional Network trained on the 59-class landmark representation.

### Configuration

```text
Input features:   150
Hidden dimension: 128
Temporal blocks:  4
Dilations:        1, 2, 4, 8
Dropout:           0.3
Output classes:   59
Parameters:       423,483
```

The final checkpoint is used by:

```text
scripts/predict_video.py
scripts/predict_raw_video.py
scripts/webcam_demo.py
```

Experimental checkpoints from development are not required for the final application.
