import numpy as np
import glob
import os


files = glob.glob(
    r".\data\processed\landmarks_test\**\*.npy",
    recursive=True
)

for file in files:
    x = np.load(file)

    left_missing = np.isnan(x[:, :21, :]).all(axis=(1, 2))
    right_missing = np.isnan(x[:, 21:42, :]).all(axis=(1, 2))

    left_pattern = "".join(
        "X" if missing else "."
        for missing in left_missing
    )

    right_pattern = "".join(
        "X" if missing else "."
        for missing in right_missing
    )

    print()
    print(os.path.basename(file))
    print("Left : ", left_pattern)
    print("Right:", right_pattern)