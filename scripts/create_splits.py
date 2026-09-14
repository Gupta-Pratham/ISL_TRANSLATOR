import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

INDEX_FILE = Path(
    r".\data\processed\dataset_index.csv"
)

OUTPUT_DIR = Path(
    r".\data\processed"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.15
VAL_SIZE = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    df = pd.read_csv(INDEX_FILE)

    print("============================================")
    print("Creating Train / Validation / Test Splits")
    print("============================================")

    print(f"Total videos: {len(df)}")
    print(f"Total classes: {df['label'].nunique()}")

    # --------------------------------------------------------
    # First split:
    #
    # 85% temporary train
    # 15% test
    # --------------------------------------------------------

    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # --------------------------------------------------------
    # Second split:
    #
    # From the remaining 85%:
    # validation should be 15% of total
    #
    # 15 / 85 = 0.17647
    # --------------------------------------------------------

    val_relative_size = VAL_SIZE / (1.0 - TEST_SIZE)

    train, val = train_test_split(
        train_val,
        test_size=val_relative_size,
        random_state=RANDOM_STATE,
        stratify=train_val["label"]
    )

    # --------------------------------------------------------
    # Reset indices
    # --------------------------------------------------------

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\nSplit sizes:")
    print(f"Train: {len(train)}")
    print(f"Val  : {len(val)}")
    print(f"Test : {len(test)}")

    print("\nTrain distribution:")
    print(
        train["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValidation distribution:")
    print(
        val["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTest distribution:")
    print(
        test["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_ids = set(train["video_id"])
    val_ids = set(val["video_id"])
    test_ids = set(test["video_id"])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    print("\nLeakage check:")

    print(
        "Train ∩ Val  :",
        len(train_val_overlap)
    )

    print(
        "Train ∩ Test :",
        len(train_test_overlap)
    )

    print(
        "Val ∩ Test   :",
        len(val_test_overlap)
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "val.csv")
    print(OUTPUT_DIR / "test.csv")

    print("============================================")


if __name__ == "__main__":
    main()