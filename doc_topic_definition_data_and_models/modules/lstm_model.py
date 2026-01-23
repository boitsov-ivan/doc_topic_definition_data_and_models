import torch


class LSTMClassifier(torch.nn.Module):
    """LSTM model for text classification"""

    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, input_size)
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
