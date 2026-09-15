from pathlib import Path
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CLEAN_DIR = BASE_DIR / "data" / "processed" / "landmarks_clean"
PREPROCESSED_DIR = BASE_DIR / "data" / "processed" / "landmarks_preprocessed"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "landmarks_masked"

EXPECTED_LANDMARKS = 50
EXPECTED_COORDINATES = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(clean_path):
    filename = clean_path.name

    # Preprocessed files are stored directly in the root
    # of landmarks_preprocessed.
    preprocessed_path = PREPROCESSED_DIR / filename

    # Preserve the clean dataset's folder structure in output.
    relative_path = clean_path.relative_to(CLEAN_DIR)
    output_path = OUTPUT_DIR / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found for {filename}: "
            f"{preprocessed_path}"
        )

    clean = np.load(clean_path).astype(np.float32)
    preprocessed = np.load(preprocessed_path).astype(np.float32)

    # --------------------------------------------------------
    # Validate clean data
    # --------------------------------------------------------

    if clean.ndim != 3:
        raise ValueError(
            f"Expected clean shape (T, 50, 3), "
            f"got {clean.shape} for {filename}"
        )

    if clean.shape[1:] != (EXPECTED_LANDMARKS, EXPECTED_COORDINATES):
        raise ValueError(
            f"Expected clean shape (T, 50, 3), "
            f"got {clean.shape} for {filename}"
        )

    # --------------------------------------------------------
    # Validate preprocessed data
    # --------------------------------------------------------

    if preprocessed.ndim != 3:
        raise ValueError(
            f"Expected preprocessed shape (T, 50, 3), "
            f"got {preprocessed.shape} for {filename}"
        )

    if preprocessed.shape != clean.shape:
        raise ValueError(
            f"Shape mismatch for {filename}: "
            f"clean={clean.shape}, "
            f"preprocessed={preprocessed.shape}"
        )

    # --------------------------------------------------------
    # Create landmark availability mask
    #
    # (T, 50, 3) -> (T, 50)
    #
    # 1 = landmark originally detected
    # 0 = landmark originally missing
    # --------------------------------------------------------

    coordinate_valid = ~np.isnan(clean)

    landmark_mask = coordinate_valid.any(axis=2).astype(np.float32)

    # --------------------------------------------------------
    # Flatten preprocessed landmarks
    #
    # (T, 50, 3) -> (T, 150)
    # --------------------------------------------------------

    positions = preprocessed.reshape(
        preprocessed.shape[0], -1
    )

    # --------------------------------------------------------
    # Combine positions + missingness mask
    #
    # 150 + 50 = 200 features/frame
    # --------------------------------------------------------

    combined = np.concatenate(
        [positions, landmark_mask],
        axis=1
    ).astype(np.float32)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if np.isnan(combined).any():
        raise ValueError(
            f"NaN detected in output for {filename}"
        )

    if np.isinf(combined).any():
        raise ValueError(
            f"Inf detected in output for {filename}"
        )

    if combined.shape[1] != 200:
        raise ValueError(
            f"Expected 200 features/frame, "
            f"got {combined.shape[1]} for {filename}"
        )

    np.save(output_path, combined)

    return clean.shape, combined.shape
    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    if clean.ndim != 3:
        raise ValueError(
            f"Expected clean data shape (T, 50, 3), "
            f"got {clean.shape} for {filename}"
        )

    if clean.shape[1:] != (EXPECTED_LANDMARKS, EXPECTED_COORDINATES):
        raise ValueError(
            f"Expected clean shape (T, 50, 3), "
            f"got {clean.shape} for {filename}"
        )

    if preprocessed.ndim != 3:
        raise ValueError(
            f"Expected preprocessed data shape (T, 50, 3), "
            f"got {preprocessed.shape} for {filename}"
        )

    if preprocessed.shape != clean.shape:
        raise ValueError(
            f"Shape mismatch for {filename}: "
            f"clean={clean.shape}, preprocessed={preprocessed.shape}"
        )

    # --------------------------------------------------------
    # Create landmark-level availability mask
    #
    # A landmark is considered available if at least one of
    # its x/y/z coordinates was originally detected.
    # --------------------------------------------------------

    coordinate_valid = ~np.isnan(clean)

    # Shape:
    # (T, 50, 3) -> (T, 50)
    landmark_mask = coordinate_valid.any(axis=2).astype(np.float32)

    # --------------------------------------------------------
    # Flatten positions
    # --------------------------------------------------------

    positions = preprocessed.reshape(preprocessed.shape[0], -1)

    # --------------------------------------------------------
    # Concatenate:
    #
    # positions:     (T, 150)
    # landmark_mask:  (T, 50)
    #
    # output:         (T, 200)
    # --------------------------------------------------------

    combined = np.concatenate(
        [positions, landmark_mask],
        axis=1
    ).astype(np.float32)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if np.isnan(combined).any():
        raise ValueError(f"NaN detected in output for {filename}")

    if np.isinf(combined).any():
        raise ValueError(f"Inf detected in output for {filename}")

    if combined.shape[1] != 200:
        raise ValueError(
            f"Expected 200 features/frame, "
            f"got {combined.shape[1]} for {filename}"
        )

    np.save(output_path, combined)

    return clean.shape, combined.shape


# ============================================================
# MAIN
# ============================================================

def main():
    files = sorted(CLEAN_DIR.rglob("*.npy"))

    print("=" * 70)
    print("LANDMARK MISSINGNESS MASK CREATION")
    print("=" * 70)

    print(f"Input clean directory : {CLEAN_DIR}")
    print(f"Input preprocessed    : {PREPROCESSED_DIR}")
    print(f"Output directory      : {OUTPUT_DIR}")
    print(f"Files found           : {len(files)}")
    print()

    if not files:
        raise FileNotFoundError(
            f"No .npy files found in {CLEAN_DIR}"
        )

    processed = 0
    failed = []

    total_landmarks = 0
    available_landmarks = 0

    for i, path in enumerate(files, start=1):

        try:
            clean_shape, output_shape = process_file(path)

            # Calculate statistics for the mask
            clean = np.load(path)

            valid = ~np.isnan(clean)
            landmark_mask = valid.any(axis=2)

            total_landmarks += landmark_mask.size
            available_landmarks += landmark_mask.sum()

            processed += 1

            if i == 1 or i % 100 == 0 or i == len(files):
                print(
                    f"[{i}/{len(files)}] "
                    f"{path.relative_to(CLEAN_DIR)} | "
                    f"{clean_shape} -> {output_shape}"
                )

        except Exception as e:
            failed.append((path.name, str(e)))
            print(f"ERROR: {path.name} -> {e}")

    print()
    print("=" * 70)
    print("CREATION COMPLETE")
    print("=" * 70)

    print(f"Processed files       : {processed}")
    print(f"Failed files          : {len(failed)}")

    if total_landmarks > 0:
        availability_rate = (
            available_landmarks / total_landmarks
        ) * 100

        print(
            f"Landmark availability : "
            f"{availability_rate:.2f}%"
        )

    if failed:
        print()
        print("FAILED FILES:")
        for filename, error in failed:
            print(f"- {filename}: {error}")

    print()
    print("Output:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()