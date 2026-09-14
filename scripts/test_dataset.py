import sys
from pathlib import Path

import torch

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.dataset import create_dataloader


TRAIN_CSV = r".\data\processed\train.csv"


def main():

    dataset, loader = create_dataloader(
        TRAIN_CSV,
        batch_size=4,
        shuffle=True
    )

    print("============================================")
    print("PyTorch Dataset Test")
    print("============================================")

    print("Dataset size:", len(dataset))

    sequences, labels, lengths, padding_mask = next(
        iter(loader)
    )

    print(
        "Sequences shape:",
        sequences.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    print(
        "Lengths:",
        lengths.tolist()
    )

    print(
        "Padding mask shape:",
        padding_mask.shape
    )

    print(
        "Contains NaN:",
        torch.isnan(sequences).any().item()
    )

    print(
        "Contains Inf:",
        torch.isinf(sequences).any().item()
    )

    print("============================================")


if __name__ == "__main__":
    main()