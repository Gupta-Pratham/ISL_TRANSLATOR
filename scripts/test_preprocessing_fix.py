import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent)
)

import numpy as np

from preprocess_landmarks import (
    interpolate_sequence,
    normalize_sequence
)


FILES = [
    Path(r".\data\processed\landmarks_clean\2. quiet\MVI_9372.npy"),
    Path(r".\data\processed\landmarks_clean\2. quiet\MVI_9373.npy"),
]


for file in FILES:

    print("\nProcessing:", file.name)

    sequence = np.load(file).astype(np.float32)

    print(
        "Before:",
        "shape=", sequence.shape,
        "NaNs=", np.isnan(sequence).sum()
    )

    sequence = interpolate_sequence(sequence)

    print(
        "After interpolation:",
        "NaNs=", np.isnan(sequence).sum()
    )

    sequence = normalize_sequence(sequence)

    print(
        "After normalization:",
        "NaNs=", np.isnan(sequence).sum(),
        "Inf=", np.isinf(sequence).sum(),
        "min=", sequence.min(),
        "max=", sequence.max()
    )