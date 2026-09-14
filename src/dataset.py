import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# ISL DATASET
# ============================================================

class ISLDataset(Dataset):

    def __init__(
        self,
        csv_path,
        landmark_dir=None
    ):

        self.data = pd.read_csv(
            csv_path
        )

        self.landmark_dir = landmark_dir

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.data
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # ----------------------------------------------------
        # Landmark filename
        # ----------------------------------------------------

        filename = f"{row['video_id']}.npy"

        # ----------------------------------------------------
        # Determine landmark directory
        # ----------------------------------------------------

        if self.landmark_dir is not None:

            landmark_path = (
                self.landmark_dir
                / filename
            )

        else:

            # Existing/default representation
            landmark_path = (
                row["landmark_path"]
            )

        # ----------------------------------------------------
        # Load landmarks
        # ----------------------------------------------------

        sequence = np.load(
            landmark_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Expected shape:
        #
        # (T, L, 3)
        #
        # Convert to:
        #
        # (T, L*3)
        # ----------------------------------------------------

        sequence = sequence.reshape(
            sequence.shape[0],
            -1
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if np.isnan(sequence).any():

            raise ValueError(
                f"NaN found in {landmark_path}"
            )

        if np.isinf(sequence).any():

            raise ValueError(
                f"Inf found in {landmark_path}"
            )

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        sequence = torch.from_numpy(
            sequence
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = int(
            row["class_id"]
        )

        return (
            sequence,
            label
        )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    sequences = [
        item[0]
        for item in batch
    ]

    labels = torch.tensor(
        [
            item[1]
            for item in batch
        ],
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Original sequence lengths
    # --------------------------------------------------------

    lengths = torch.tensor(
        [
            sequence.shape[0]
            for sequence in sequences
        ],
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Pad variable-length sequences
    # --------------------------------------------------------

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    # --------------------------------------------------------
    # Padding mask
    #
    # False = real frame
    # True  = padding
    # --------------------------------------------------------

    batch_size = padded_sequences.shape[0]

    max_length = padded_sequences.shape[1]

    padding_mask = torch.ones(
        batch_size,
        max_length,
        dtype=torch.bool
    )

    for i, length in enumerate(
        lengths
    ):

        padding_mask[
            i,
            :length
        ] = False

    return (
        padded_sequences,
        labels,
        lengths,
        padding_mask
    )


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(
    csv_path,
    batch_size=8,
    shuffle=False,
    landmark_dir=None
):

    dataset = ISLDataset(
        csv_path,
        landmark_dir=landmark_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return (
        dataset,
        loader
    )