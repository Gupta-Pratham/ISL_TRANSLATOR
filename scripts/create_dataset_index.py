import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

VIDEO_DIR = Path(
    r".\data\raw\INCLUDE\Adjectives"
)

DATA_DIR = Path(
    r".\data\processed\landmarks_preprocessed"
)

OUTPUT_FILE = Path(
    r".\data\processed\dataset_index.csv"
)


# ============================================================
# BUILD VIDEO → LABEL MAPPING
# ============================================================

def build_video_label_map():

    mapping = {}

    video_files = sorted(
        VIDEO_DIR.rglob("*.MOV")
    )

    for video in video_files:

        relative_parts = video.relative_to(VIDEO_DIR).parts

        if len(relative_parts) >= 2:
            label = relative_parts[0]
        else:
            label = video.parent.name

        mapping[video.stem] = label

    return mapping

# ============================================================
# MAIN
# ============================================================

def main():

    video_label_map = build_video_label_map()

    npy_files = sorted(
        DATA_DIR.glob("*.npy")
    )

    if not npy_files:
        print("ERROR: No preprocessed .npy files found.")
        return

    records = []

    missing_labels = []

    for file in npy_files:

        video_id = file.stem

        if video_id not in video_label_map:

            missing_labels.append(
                video_id
            )

            continue

        label = video_label_map[video_id]

        sequence = np.load(
            file,
            mmap_mode="r"
        )

        records.append(
            {
                "video_id": video_id,
                "video_path": str(file),
                "label": label,
                "num_frames": sequence.shape[0],
                "num_landmarks": sequence.shape[1],
                "num_coordinates": sequence.shape[2],
            }
        )

    if missing_labels:

        print("WARNING: Some files have no matching video:")
        print(missing_labels)

    df = pd.DataFrame(records)

    if df.empty:

        print("ERROR: No matching videos found.")
        return

    # --------------------------------------------------------
    # Assign class IDs
    # --------------------------------------------------------

    labels = sorted(
        df["label"].unique()
    )

    label_to_id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    df["class_id"] = df["label"].map(
        label_to_id
    )

    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    df = df[
        [
            "video_id",
            "video_path",
            "label",
            "class_id",
            "num_frames",
            "num_landmarks",
            "num_coordinates",
        ]
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("============================================")
    print("ISL Dataset Index")
    print("============================================")

    print(f"Processed files: {len(npy_files)}")
    print(f"Matched files  : {len(df)}")
    print(f"Classes        : {df['label'].nunique()}")
    print(f"Frames         : {df['num_frames'].sum()}")

    print("\nClass distribution:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFrame statistics:")
    print(
        df["num_frames"].describe()
    )

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("============================================")


if __name__ == "__main__":
    main()