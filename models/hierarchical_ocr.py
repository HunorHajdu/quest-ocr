"""
Hierarchical OCR Model
"""

import torch
import torch.nn as nn
from quest_ocr.config import NUM_CHARS, MAX_VOCAB_SIZE, MAX_NGRAM_VOCAB_SIZE


class HierarchicalOCRModel(nn.Module):
    def __init__(
        self,
        input_channels=1,
        hidden_size=256,
        num_chars=NUM_CHARS,
        num_words=MAX_VOCAB_SIZE,
        num_ngrams=MAX_NGRAM_VOCAB_SIZE,
        architecture="lstm",
        num_layers=2,
        dropout=0.2,
    ):
        """
        Hierarchical OCR model with multi-level prediction without language classification

        Args:
            input_channels: Number of input channels (1 for grayscale)
            hidden_size: Hidden size for sequence models
            num_chars: Number of characters in the character set
            num_words: Size of word vocabulary
            num_ngrams: Size of n-gram vocabulary
            architecture: Choice of sequence model ('rnn', 'gru', 'lstm', or 'transformer')
            num_layers: Number of layers in the sequence model
            dropout: Dropout rate between layers
        """
        super(HierarchicalOCRModel, self).__init__()

        self.architecture = architecture.lower()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

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
        if self.architecture in ["rnn", "gru", "lstm"]:
            # Base sequence model for shared features
            if self.architecture == "lstm":
                self.seq_model = nn.LSTM(
                    input_size=self.seq_input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True,
                    batch_first=True,
                )
            elif self.architecture == "gru":
                self.seq_model = nn.GRU(
                    input_size=self.seq_input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True,
                    batch_first=True,
                )
            else:  # rnn
                self.seq_model = nn.RNN(
                    input_size=self.seq_input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True,
                    batch_first=True,
                )

            # Output size for bidirectional model
            self.seq_output_size = hidden_size * 2

            # Character-level decoder (without language feedback)
            self.char_decoder = nn.Sequential(
                nn.Linear(self.seq_output_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_chars),
            )

            # N-gram level decoder with character feedback
            self.ngram_decoder = nn.Sequential(
                nn.Linear(self.seq_output_size + hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_ngrams),
            )

            # Word-level decoder with character and n-gram feedback
            self.word_decoder = nn.Sequential(
                nn.Linear(self.seq_output_size + hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_words),
            )

            # Embeddings for feedback
            self.char_embedding = nn.Embedding(num_chars, hidden_size)
            self.ngram_embedding = nn.Embedding(num_ngrams, hidden_size)
            self.word_embedding = nn.Embedding(num_words, hidden_size)

            # Attention mechanisms for bidirectional feedback
            self.char_attention = nn.Sequential(
                nn.Linear(self.seq_output_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

            self.ngram_attention = nn.Sequential(
                nn.Linear(self.seq_output_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

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
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=num_layers
            )

            self.seq_output_size = hidden_size

            # Multi-head attention for hierarchical context modeling
            self.multihead_attention = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=8, dropout=dropout, batch_first=True
            )

            # Character-level decoder (without language feedback)
            self.char_decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_chars),
            )

            # N-gram level decoder
            self.ngram_decoder = nn.Sequential(
                nn.Linear(hidden_size + hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_ngrams),
            )

            # Word-level decoder
            self.word_decoder = nn.Sequential(
                nn.Linear(hidden_size + hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_words),
            )

            # Embeddings for feedback
            self.char_embedding = nn.Embedding(num_chars, hidden_size)
            self.ngram_embedding = nn.Embedding(num_ngrams, hidden_size)
            self.word_embedding = nn.Embedding(num_words, hidden_size)

            # Fusion layers
            self.char_fusion = nn.Linear(hidden_size * 2, hidden_size)
            self.ngram_fusion = nn.Linear(hidden_size * 2, hidden_size)
            self.word_fusion = nn.Linear(hidden_size * 2, hidden_size)

        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Choose from 'rnn', 'gru', 'lstm', or 'transformer'."
            )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN feature extraction (batch, channels, height, width)
        conv_features = self.cnn(x)
        _, channels, height, width = conv_features.size()

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
                    self.hidden_size,
                    device=conv_features.device,
                )
                c0 = torch.zeros(
                    self.num_layers * num_directions,
                    batch_size,
                    self.hidden_size,
                    device=conv_features.device,
                )
                seq_output, _ = self.seq_model(conv_features, (h0, c0))
            else:
                # RNN and GRU only need h0
                h0 = torch.zeros(
                    self.num_layers * num_directions,
                    batch_size,
                    self.hidden_size,
                    device=conv_features.device,
                )
                seq_output, _ = self.seq_model(conv_features, h0)

            # Apply attention for character decoding
            char_attention_weights = torch.softmax(
                self.char_attention(seq_output).squeeze(-1), dim=1
            )
            char_context = torch.bmm(
                char_attention_weights.unsqueeze(1), seq_output
            ).squeeze(1)  # (batch, hidden*2)

            # Apply attention for n-gram decoding
            ngram_attention_weights = torch.softmax(
                self.ngram_attention(seq_output).squeeze(-1), dim=1
            )
            ngram_context = torch.bmm(
                ngram_attention_weights.unsqueeze(1), seq_output
            ).squeeze(1)  # (batch, hidden*2)

            # Hierarchical decoding with bidirectional feedback
            # Character-level decoding (without language feedback)
            char_logits = self.char_decoder(seq_output)

            # Get character embeddings for feedback to other levels
            char_preds = torch.argmax(char_logits, dim=2)
            char_embeddings = self.char_embedding(
                char_preds
            )  # (batch, seq_len, hidden)
            char_feature = char_embeddings.mean(dim=1)  # (batch, hidden)

            # N-gram level with character feedback
            ngram_features = torch.cat([seq_output, char_embeddings], dim=2)
            ngram_logits = self.ngram_decoder(ngram_features)

            # Get n-gram embeddings for feedback to word level
            ngram_preds = torch.argmax(ngram_logits, dim=2)
            ngram_embeddings = self.ngram_embedding(
                ngram_preds
            )  # (batch, seq_len, hidden)
            ngram_feature = ngram_embeddings.mean(dim=1)  # (batch, hidden)

            # Word level with character and n-gram feedback
            word_features = torch.cat(
                [seq_output, char_embeddings, ngram_embeddings], dim=2
            )
            word_logits = self.word_decoder(word_features)

        elif self.architecture == "transformer":
            # Project to transformer dimension
            transformer_input = self.input_projection(conv_features)

            # Add positional encoding
            seq_len = transformer_input.size(1)
            pos_enc = self.pos_encoder[:, :seq_len, :]
            transformer_input = transformer_input + pos_enc

            # Process with transformer
            seq_output = self.transformer_encoder(transformer_input)

            # Character-level decoding (without language feedback)
            char_query = seq_output
            char_context, _ = self.multihead_attention(
                query=char_query, key=seq_output, value=seq_output
            )

            # Character-level decoding
            char_logits = self.char_decoder(char_context)

            # Get character predictions for feedback
            char_preds = torch.argmax(char_logits, dim=2)
            char_embeddings = self.char_embedding(char_preds)

            # Fuse character information with sequence features
            char_fused = self.char_fusion(
                torch.cat([seq_output, char_embeddings], dim=2)
            )

            # N-gram level with character feedback
            ngram_context, _ = self.multihead_attention(
                query=char_fused, key=seq_output, value=seq_output
            )

            ngram_features = torch.cat([ngram_context, char_embeddings], dim=2)
            ngram_logits = self.ngram_decoder(ngram_features)

            # Get n-gram predictions for feedback
            ngram_preds = torch.argmax(ngram_logits, dim=2)
            ngram_embeddings = self.ngram_embedding(ngram_preds)

            # Fuse n-gram information
            ngram_fused = self.ngram_fusion(
                torch.cat([char_fused, ngram_embeddings], dim=2)
            )

            # Word level with character and n-gram feedback
            word_context, _ = self.multihead_attention(
                query=ngram_fused, key=seq_output, value=seq_output
            )

            word_features = torch.cat(
                [word_context, char_embeddings, ngram_embeddings], dim=2
            )
            word_logits = self.word_decoder(word_features)

        # Return log probabilities for each level
        return {
            "char": nn.functional.log_softmax(char_logits, dim=2),
            "ngram": nn.functional.log_softmax(ngram_logits, dim=2),
            "word": nn.functional.log_softmax(word_logits, dim=2),
        }
