"""
Simple OCR Model
"""

import torch
import torch.nn as nn
from config import NUM_CHARS


class OCRModel(nn.Module):
    def __init__(
        self,
        input_channels=1,
        hidden_size=128,
        num_classes=NUM_CHARS,
        architecture="lstm",
        num_layers=2,
        dropout=0.2,
    ):
        """
        OCR model with configurable architecture

        Args:
            input_channels: Number of input channels (1 for grayscale)
            hidden_size: Hidden size for sequence models
            num_classes: Number of characters in the character set
            architecture: Choice of sequence model ('rnn', 'gru', 'lstm', or 'transformer')
            num_layers: Number of layers in the sequence model
            dropout: Dropout rate between layers
        """
        super(OCRModel, self).__init__()

        self.architecture = architecture.lower()
        self.num_layers = num_layers

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        # Calculate CNN output size
        self.seq_input_size = 128 * 16  # 128 channels * 16 height

        # Choose sequence model architecture
        if self.architecture == "rnn":
            self.seq_model = nn.RNN(
                input_size=self.seq_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
                batch_first=True,
            )
            self.output_size = hidden_size * 2  # *2 for bidirectional

        elif self.architecture == "gru":
            self.seq_model = nn.GRU(
                input_size=self.seq_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
                batch_first=True,
            )
            self.output_size = hidden_size * 2  # *2 for bidirectional

        elif self.architecture == "lstm":
            self.seq_model = nn.LSTM(
                input_size=self.seq_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
                batch_first=True,
            )
            self.output_size = hidden_size * 2  # *2 for bidirectional

        elif self.architecture == "transformer":
            # Project input to transformer dimension
            self.input_projection = nn.Linear(self.seq_input_size, hidden_size)

            # Position encoding for transformer
            max_seq_length = 100
            self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, hidden_size))
            nn.init.xavier_uniform_(self.pos_encoder)

            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=32,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=num_layers
            )
            self.output_size = hidden_size

        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Choose from 'rnn', 'gru', 'lstm', or 'transformer'."
            )

        # Output layer
        self.fc = nn.Linear(self.output_size, num_classes)

    def forward(self, x):
        # CNN feature extraction (batch, channels, height, width)
        conv_features = self.cnn(x)
        batch_size, channels, height, width = conv_features.size()

        # Reshape for sequence model: (batch, sequence_length, features)
        conv_features = conv_features.permute(
            0, 3, 1, 2
        )  # (batch, width, channels, height)
        conv_features = conv_features.reshape(batch_size, width, channels * height)

        # Process with the chosen sequence model
        if self.architecture in ["lstm", "gru", "rnn"]:
            # Initialize hidden states
            batch_size = conv_features.size(0)
            num_directions = 2  # bidirectional

            if self.architecture == "lstm":
                # LSTM needs both h0 and c0
                h0 = torch.zeros(
                    self.num_layers * num_directions,
                    batch_size,
                    self.output_size // num_directions,
                    device=conv_features.device,
                )
                c0 = torch.zeros(
                    self.num_layers * num_directions,
                    batch_size,
                    self.output_size // num_directions,
                    device=conv_features.device,
                )
                seq_output, _ = self.seq_model(conv_features, (h0, c0))
            else:
                # RNN and GRU only need h0
                h0 = torch.zeros(
                    self.num_layers * num_directions,
                    batch_size,
                    self.output_size // num_directions,
                    device=conv_features.device,
                )
                seq_output, _ = self.seq_model(conv_features, h0)

        elif self.architecture == "transformer":
            # Project to transformer dimension
            transformer_input = self.input_projection(conv_features)

            # Add positional encoding
            seq_len = transformer_input.size(1)
            pos_enc = self.pos_encoder[:, :seq_len, :]
            transformer_input = transformer_input + pos_enc

            # Process with transformer
            seq_output = self.transformer_encoder(transformer_input)

        # Apply output layer to each time step
        output = self.fc(seq_output)

        # Return log probabilities
        return nn.functional.log_softmax(output, dim=2)