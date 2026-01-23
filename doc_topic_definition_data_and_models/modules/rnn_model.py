import torch


class RNNClassifier(torch.nn.Module):
    """RNN model for text classification"""

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
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity="relu",
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        outputs, (hn, cn) = self.rnn(x)
        last_output = outputs[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output)
