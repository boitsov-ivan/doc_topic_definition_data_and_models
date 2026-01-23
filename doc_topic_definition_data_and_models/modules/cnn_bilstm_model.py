from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTMClassifier(nn.Module):
    """CNN-BiLSTM hybrid model for text classification"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        num_filters: int = 100,
        filter_sizes: Tuple[int, ...] = (3, 4, 5),
        bidirectional: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )

        self.lstm = nn.LSTM(
            input_size=num_filters * len(filter_sizes),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2 if bidirectional else hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.embedding.padding_idx])

        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)

        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "bias_hh" in name or "bias_ih" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, return_attention: bool = False):
        """
        Args:
            x: [batch_size, seq_len]
            return_attention: возвращать ли веса внимания

        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, seq_len] (опционально)
        """
        batch_size, seq_len = x.size()

        embedded = self.embedding(x)

        embedded = embedded.unsqueeze(1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            conv_out = conv_out.squeeze(3)
            pooled = F.max_pool1d(conv_out, kernel_size=int(conv_out.size(2))).squeeze(
                2
            )
            # pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        cnn_features = torch.cat(conv_outputs, dim=1)

        lstm_input = cnn_features.unsqueeze(1).repeat(1, seq_len, 1)

        lstm_output, (hidden, cell) = self.lstm(lstm_input)

        attention_scores = self.attention(lstm_output).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)

        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(
            1
        )

        context_vector = self.dropout(context_vector)
        logits = self.fc(context_vector)

        if return_attention:
            return logits, attention_weights
        return logits
