from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "tcn_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def main():

    print("=" * 70)
    print("TCN ERROR ANALYSIS")
    print("=" * 70)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "sample_index",
        "actual_class_id",
        "actual_label",
        "predicted_class_id",
        "predicted_label",
        "correct",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(f"Test samples: {len(df)}")

    # Make sure correct is boolean
    df["correct"] = df["correct"].astype(bool)

    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    total = len(df)
    correct = int(df["correct"].sum())
    incorrect = total - correct

    accuracy = correct / total

    print()
    print("-" * 70)
    print("OVERALL PERFORMANCE")
    print("-" * 70)

    print(f"Total samples     : {total}")
    print(f"Correct           : {correct}")
    print(f"Incorrect         : {incorrect}")
    print(f"Accuracy          : {accuracy * 100:.2f}%")

    # ========================================================
    # INCORRECT PREDICTIONS
    # ========================================================

    errors = df[~df["correct"]].copy()

    print()
    print("-" * 70)
    print("ERROR SUMMARY")
    print("-" * 70)

    print(f"Total errors      : {len(errors)}")

    # ========================================================
    # PER-CLASS PERFORMANCE
    # ========================================================

    class_rows = []

    for class_id, group in df.groupby(
        ["actual_class_id", "actual_label"],
        sort=True
    ):

        class_id_value, class_label = class_id

        support = len(group)
        correct_count = int(group["correct"].sum())
        error_count = support - correct_count

        recall = correct_count / support

        class_rows.append(
            {
                "class_id": class_id_value,
                "label": class_label,
                "support": support,
                "correct": correct_count,
                "errors": error_count,
                "recall": recall,
                "recall_percent": recall * 100,
            }
        )

    class_df = pd.DataFrame(class_rows)

    class_df = class_df.sort_values(
        by=["recall", "support"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save complete class analysis
    # --------------------------------------------------------

    class_output = OUTPUT_DIR / "tcn_per_class_error_analysis.csv"

    class_df.to_csv(
        class_output,
        index=False
    )

    # ========================================================
    # WORST CLASSES
    # ========================================================

    print()
    print("-" * 70)
    print("10 WORST-PERFORMING CLASSES")
    print("-" * 70)

    worst_classes = class_df.head(10)

    print(
        worst_classes[
            [
                "class_id",
                "label",
                "support",
                "correct",
                "errors",
                "recall_percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "recall_percent": lambda x: f"{x:.2f}"
            }
        )
    )

    # ========================================================
    # BEST CLASSES
    # ========================================================

    print()
    print("-" * 70)
    print("10 BEST-PERFORMING CLASSES")
    print("-" * 70)

    best_classes = class_df.sort_values(
        by=["recall", "support"],
        ascending=[False, False]
    ).head(10)

    print(
        best_classes[
            [
                "class_id",
                "label",
                "support",
                "correct",
                "errors",
                "recall_percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "recall_percent": lambda x: f"{x:.2f}"
            }
        )
    )

    # ========================================================
    # CONFUSION PAIRS
    # ========================================================

    print()
    print("-" * 70)
    print("MOST FREQUENT CONFUSION PAIRS")
    print("-" * 70)

    if len(errors) == 0:

        print("No errors found.")

    else:

        confusion_pairs = (
            errors.groupby(
                [
                    "actual_class_id",
                    "actual_label",
                    "predicted_class_id",
                    "predicted_label",
                ]
            )
            .size()
            .reset_index(name="count")
            .sort_values(
                by="count",
                ascending=False
            )
            .reset_index(drop=True)
        )

        confusion_output = (
            OUTPUT_DIR
            / "tcn_confusion_pairs.csv"
        )

        confusion_pairs.to_csv(
            confusion_output,
            index=False
        )

        print(
            confusion_pairs.head(20).to_string(
                index=False
            )
        )

    # ========================================================
    # ERROR RATE BY CLASS
    # ========================================================

    class_df["error_rate"] = (
        class_df["errors"] / class_df["support"]
    )

    # ========================================================
    # LOW-SUPPORT CLASSES
    # ========================================================

    low_support = class_df[
        class_df["support"] <= 2
    ].copy()

    print()
    print("-" * 70)
    print("LOW-SUPPORT CLASSES")
    print("-" * 70)

    print(
        f"Classes with <= 2 test samples: {len(low_support)}"
    )

    if len(low_support) > 0:
        print(
            low_support[
                [
                    "class_id",
                    "label",
                    "support",
                    "correct",
                    "errors",
                    "recall_percent",
                ]
            ].to_string(
                index=False,
                formatters={
                    "recall_percent": lambda x: f"{x:.2f}"
                }
            )
        )

    # ========================================================
    # ERROR DISTRIBUTION
    # ========================================================

    error_counts = Counter(
        errors["actual_label"]
    )

    print()
    print("-" * 70)
    print("CLASSES CONTRIBUTING MOST TO TOTAL ERRORS")
    print("-" * 70)

    if error_counts:

        error_distribution = pd.DataFrame(
            error_counts.items(),
            columns=["label", "error_count"]
        ).sort_values(
            by="error_count",
            ascending=False
        )

        print(
            error_distribution.head(10).to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE ALL ERRORS
    # ========================================================

    errors_output = (
        OUTPUT_DIR
        / "tcn_errors_only.csv"
    )

    errors.to_csv(
        errors_output,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print("Generated files:")

    print(f"1. {class_output}")
    print(f"2. {errors_output}")

    if len(errors) > 0:
        print(f"3. {OUTPUT_DIR / 'tcn_confusion_pairs.csv'}")

    print()


if __name__ == "__main__":
    main()