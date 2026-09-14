import torch
import torch.nn as nn


# ============================================================
# LSTM BASELINE
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=150,
        hidden_size=128,
        num_layers=2,
        num_classes=8,
        dropout=0.3
    ):

        super().__init__()

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size
            ),
            nn.LayerNorm(
                hidden_size
            ),
            nn.GELU()
        )

        # ----------------------------------------------------
        # Bidirectional LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        padding_mask
    ):

        # x:
        # (B, T, 150)

        x = self.input_projection(x)

        # x:
        # (B, T, 128)

        lstm_output, _ = self.lstm(x)

        # lstm_output:
        # (B, T, 256)

        # ----------------------------------------------------
        # Mask padded frames
        # ----------------------------------------------------

        valid_mask = ~padding_mask

        mask = valid_mask.unsqueeze(-1)

        masked_output = (
            lstm_output * mask
        )

        # ----------------------------------------------------
        # Mean pooling over valid frames
        # ----------------------------------------------------

        lengths = valid_mask.sum(
            dim=1,
            keepdim=True
        ).clamp(
            min=1
        )

        pooled = (
            masked_output.sum(dim=1)
            / lengths
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits

# ============================================================
# GRU BASELINE
# ============================================================

class GRUClassifier(nn.Module):

    def __init__(
        self,
        input_size=150,
        hidden_size=128,
        num_layers=2,
        num_classes=8,
        dropout=0.3
    ):

        super().__init__()

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size
            ),
            nn.LayerNorm(
                hidden_size
            ),
            nn.GELU()
        )

        # ----------------------------------------------------
        # Bidirectional GRU
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        padding_mask
    ):

        # x:
        # (B, T, 150)

        x = self.input_projection(x)

        # x:
        # (B, T, 128)

        gru_output, _ = self.gru(x)

        # gru_output:
        # (B, T, 256)

        # ----------------------------------------------------
        # Mask padded frames
        # ----------------------------------------------------

        valid_mask = ~padding_mask

        mask = valid_mask.unsqueeze(-1)

        masked_output = (
            gru_output * mask
        )

        # ----------------------------------------------------
        # Mean pooling over valid frames
        # ----------------------------------------------------

        lengths = valid_mask.sum(
            dim=1,
            keepdim=True
        ).clamp(
            min=1
        )

        pooled = (
            masked_output.sum(dim=1)
            / lengths
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits

# ============================================================
# TCN BASELINE
# ============================================================

class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.3
    ):

        super().__init__()

        padding = (
            (kernel_size - 1)
            * dilation
        )

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.norm1 = nn.BatchNorm1d(
            out_channels
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.norm2 = nn.BatchNorm1d(
            out_channels
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        if in_channels != out_channels:

            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.residual = nn.Identity()

    def forward(self, x):

        # x:
        # (B, C, T)

        residual = self.residual(
            x
        )

        out = self.conv1(
            x
        )

        # Remove extra right padding
        out = out[:, :, :x.size(2)]

        out = self.norm1(
            out
        )

        out = self.relu(
            out
        )

        out = self.dropout(
            out
        )

        out = self.conv2(
            out
        )

        # Remove extra right padding
        out = out[:, :, :x.size(2)]

        out = self.norm2(
            out
        )

        out = self.relu(
            out
        )

        out = self.dropout(
            out
        )

        return self.relu(
            out + residual
        )


# ============================================================
# TCN CLASSIFIER
# ============================================================

class TCNClassifier(nn.Module):

    def __init__(
        self,
        input_size=150,
        hidden_size=128,
        num_classes=8,
        dropout=0.3
    ):

        super().__init__()

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size
            ),
            nn.LayerNorm(
                hidden_size
            ),
            nn.GELU()
        )

        # ----------------------------------------------------
        # Temporal convolution blocks
        #
        # Dilations:
        # 1, 2, 4, 8
        #
        # This gives the network an expanding temporal
        # receptive field.
        # ----------------------------------------------------

        self.tcn = nn.Sequential(

            TemporalBlock(
                hidden_size,
                hidden_size,
                kernel_size=3,
                dilation=1,
                dropout=dropout
            ),

            TemporalBlock(
                hidden_size,
                hidden_size,
                kernel_size=3,
                dilation=2,
                dropout=dropout
            ),

            TemporalBlock(
                hidden_size,
                hidden_size,
                kernel_size=3,
                dilation=4,
                dropout=dropout
            ),

            TemporalBlock(
                hidden_size,
                hidden_size,
                kernel_size=3,
                dilation=8,
                dropout=dropout
            )
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        padding_mask
    ):

        # x:
        # (B, T, 150)

        x = self.input_projection(
            x
        )

        # x:
        # (B, T, 128)

        # ----------------------------------------------------
        # Convert to Conv1D format
        #
        # (B, T, C)
        #       ↓
        # (B, C, T)
        # ----------------------------------------------------

        x = x.transpose(
            1,
            2
        )

        # ----------------------------------------------------
        # TCN
        # ----------------------------------------------------

        x = self.tcn(
            x
        )

        # x:
        # (B, 128, T)

        # ----------------------------------------------------
        # Convert back
        # ----------------------------------------------------

        x = x.transpose(
            1,
            2
        )

        # x:
        # (B, T, 128)

        # ----------------------------------------------------
        # Mask padded frames
        # ----------------------------------------------------

        valid_mask = ~padding_mask

        mask = valid_mask.unsqueeze(
            -1
        )

        x = x * mask

        # ----------------------------------------------------
        # Mean pooling over valid frames
        # ----------------------------------------------------

        lengths = valid_mask.sum(
            dim=1,
            keepdim=True
        ).clamp(
            min=1
        )

        pooled = (
            x.sum(dim=1)
            / lengths
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TRANSFORMER BASELINE
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size=150,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=8,
        dropout=0.3,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=512,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            d_model,
            num_classes,
        )

    def forward(self, x, padding_mask=None):

        # x:
        # [batch, time, features]

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        # Masked mean pooling
        if padding_mask is not None:

            valid_mask = (
                ~padding_mask
            ).unsqueeze(-1)

            x = x.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0
            )

            lengths = valid_mask.sum(
                dim=1
            ).clamp(min=1)

            x = x.sum(dim=1) / lengths

        else:

            x = x.mean(dim=1)

        x = self.dropout(x)

        return self.classifier(x)
