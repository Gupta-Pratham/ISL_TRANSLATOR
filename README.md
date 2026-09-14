# Signer-Independent Indian Sign Language Recognition and Translation

A research-oriented final-year project investigating efficient spatio-temporal deep learning for Indian Sign Language (ISL) recognition and translation.

## Project Status

🚧 Active Research / Development

The current repository contains the initial landmark-based recognition pipeline and preliminary experiments on a small subset of the INCLUDE dataset.

The current results are prototype results and should not be interpreted as final generalization performance.

---

## Research Direction

### Research Question

Can lightweight spatio-temporal landmark representations provide accurate, signer-independent Indian Sign Language recognition while maintaining sufficiently low inference latency for real-time use?

### Hypothesis

A carefully designed spatio-temporal representation using hand and upper-body pose landmarks, combined with efficient temporal models, can provide a useful accuracy-latency-generalization trade-off for real-time ISL recognition.

---

## Current Pipeline

```text
ISL Video
    ↓
MediaPipe Hand + Pose Landmark Extraction
    ↓
Missing Landmark Handling
    ↓
Temporal Interpolation
    ↓
Shoulder-Based Normalization
    ↓
Variable-Length Sequence
    ↓
Temporal Deep Learning Model
    ↓
ISL Class Prediction