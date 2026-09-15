# Dataset

## INCLUDE Dataset

This project uses the **INCLUDE (Indian Sign Language Dataset)** developed by AI4Bharat.

The complete dataset is not included in this repository because of its large size.

## Dataset Used

The final experiment uses the **Adjectives** portion of INCLUDE.

### Statistics

| Property | Value |
|---|---:|
| Total videos | 791 |
| Classes | 59 |
| Total frames | 46,619 |
| Training videos | 553 |
| Validation videos | 119 |
| Test videos | 119 |

## Expected Directory

After downloading and extracting the dataset, the project expects:

```text
data/
└── raw/
    └── INCLUDE/
        └── Adjectives/
            ├── <class folders>/
            │   └── *.MOV
            └── ...
```

Generated landmark data is stored under:

```text
data/processed/
```

These generated files are not required for teammates who only want to run the already-trained inference application.

## Reproducing the Pipeline

With the dataset in the expected location:

```powershell
python scripts\extract_landmarks.py
python scripts\preprocess_landmarks.py
python scripts	rain_tcn.py
```

The final trained model is:

```text
models/checkpoints/tcn_baseline_best.pt
```

## Evaluation Note

The metadata used in this implementation did not provide signer IDs.

Therefore, the reported evaluation is **video-disjoint**, not a verified signer-independent evaluation.
