"""BiLSTM + CTC Decoder for sequential text recognition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from medscript.utils.logging import get_logger

logger = get_logger(__name__)


class BiLSTMCTCDecoder(nn.Module):
    """
    Bidirectional LSTM with CTC decoder for handwriting recognition.

    Takes sequential features from the vision encoder and produces
    character-level predictions. The BiLSTM captures temporal context
    in both directions, which is critical for cursive handwriting where
    character shapes depend on neighboring characters.

    Architecture:
        Sequential features (B, T, input_dim)
        → BiLSTM layers
        → Linear projection to vocab_size
        → log_softmax
        → CTC decoding (greedy or beam search)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        vocab_size: int = 95,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension (from vision encoder)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            vocab_size: Output vocabulary size (including blank token)
            dropout: Dropout probability between LSTM layers
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection (optional normalization before LSTM)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output projection: LSTM output → vocab logits
        lstm_output_dim = hidden_size * self.num_directions
        self.output_proj = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, vocab_size),
        )

        logger.info(
            "bilstm_ctc_initialized",
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            bidirectional=bidirectional,
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through BiLSTM.

        Args:
            features: (B, T, input_dim) sequential features from encoder
            feature_lengths: (B,) actual sequence lengths (for packed sequences)

        Returns:
            log_probs: (B, T, vocab_size) log probabilities for CTC loss
        """
        # Input projection
        x = self.input_proj(features)

        # Pack sequences if lengths provided (for efficiency)
        if feature_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, feature_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Unpack if needed
        if feature_lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Output projection → log probabilities
        logits = self.output_proj(lstm_out)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def greedy_decode(self, log_probs: torch.Tensor) -> list[list[int]]:
        """
        Greedy CTC decoding — select most probable character at each timestep.

        Args:
            log_probs: (B, T, vocab_size) log probabilities

        Returns:
            List of decoded index sequences (one per batch item)
        """
        # Get best characters at each timestep
        best_indices = log_probs.argmax(dim=-1)  # (B, T)

        decoded_batch = []
        for seq in best_indices:
            # CTC collapse: remove consecutive duplicates and blanks
            decoded = []
            prev_idx = -1
            for idx in seq.tolist():
                if idx != 0 and idx != prev_idx:  # 0 = <blank>
                    decoded.append(idx)
                prev_idx = idx
            decoded_batch.append(decoded)

        return decoded_batch

    def get_confidence_scores(self, log_probs: torch.Tensor) -> list[list[float]]:
        """
        Extract per-character confidence scores from log probabilities.

        Args:
            log_probs: (B, T, vocab_size) log probabilities

        Returns:
            List of confidence score lists (one per batch item)
        """
        probs = log_probs.exp()  # Convert from log space
        best_probs, best_indices = probs.max(dim=-1)  # (B, T)

        confidence_batch = []
        for seq_probs, seq_indices in zip(best_probs, best_indices):
            confidences = []
            prev_idx = -1
            for prob, idx in zip(seq_probs.tolist(), seq_indices.tolist()):
                if idx != 0 and idx != prev_idx:  # Non-blank, non-duplicate
                    confidences.append(prob)
                prev_idx = idx
            confidence_batch.append(confidences)

        return confidence_batch
