import sys
from pathlib import Path

import torch

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.dataset import create_dataloader
from src.models import LSTMClassifier


TRAIN_CSV = r".\data\processed\train.csv"


def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("============================================")
    print("LSTM Model Test")
    print("============================================")

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    _, loader = create_dataloader(
        TRAIN_CSV,
        batch_size=4,
        shuffle=True
    )

    sequences, labels, lengths, padding_mask = next(
        iter(loader)
    )

    sequences = sequences.to(device)
    padding_mask = padding_mask.to(device)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LSTMClassifier(
        input_size=150,
        hidden_size=128,
        num_layers=2,
        num_classes=8,
        dropout=0.3
    ).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(
        sequences,
        padding_mask
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        "Input shape:",
        sequences.shape
    )

    print(
        "Output shape:",
        logits.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print(
        "Output contains NaN:",
        torch.isnan(logits).any().item()
    )

    print(
        "Output contains Inf:",
        torch.isinf(logits).any().item()
    )

    print("============================================")


if __name__ == "__main__":
    main()