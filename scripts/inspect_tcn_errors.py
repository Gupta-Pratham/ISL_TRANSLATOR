from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "tcn_predictions.csv"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tcn_error_video_mapping.csv"
)


def main():

    print("=" * 70)
    print("TCN ERROR → VIDEO MAPPING")
    print("=" * 70)

    predictions = pd.read_csv(PREDICTIONS_FILE)
    test = pd.read_csv(TEST_CSV)

    print(f"Prediction rows : {len(predictions)}")
    print(f"Test rows       : {len(test)}")

    # --------------------------------------------------------
    # Inspect the available test CSV columns
    # --------------------------------------------------------

    print()
    print("Test CSV columns:")
    print(test.columns.tolist())

    # --------------------------------------------------------
    # Merge using sample index if available
    # --------------------------------------------------------

    if "sample_index" in test.columns:

        merged = predictions.merge(
            test,
            on="sample_index",
            how="left",
            suffixes=("_prediction", "_test")
        )

    else:

        # Fallback: prediction sample_index appears to be
        # 1-based while dataframe index is 0-based.
        test = test.copy()
        test["sample_index"] = range(1, len(test) + 1)

        merged = predictions.merge(
            test,
            on="sample_index",
            how="left",
            suffixes=("_prediction", "_test")
        )

    # --------------------------------------------------------
    # Keep only mistakes
    # --------------------------------------------------------

    errors = merged[
        merged["correct"] == False
    ].copy()

    print()
    print(f"Total errors: {len(errors)}")

    # --------------------------------------------------------
    # Display useful columns
    # --------------------------------------------------------

    preferred_columns = [
        "sample_index",
        "actual_class_id",
        "actual_label",
        "predicted_class_id",
        "predicted_label",
        "video_id",
        "video_path",
        "landmark_path",
    ]

    available_columns = [
        col for col in preferred_columns
        if col in errors.columns
    ]

    print()
    print("-" * 70)
    print("MISCLASSIFIED VIDEOS")
    print("-" * 70)

    print(
        errors[available_columns].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    errors[available_columns].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()