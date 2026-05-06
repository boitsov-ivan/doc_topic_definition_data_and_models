import torch


class RNNClassifier(torch.nn.Module):
    """RNN model for multi-label text classification"""

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
        self.embedding = torch.nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity="relu",
            bidirectional=True, 
        )

        self.dropout = torch.nn.Dropout(dropout)
        
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        
        x = self.embedding(x)
        
        outputs, hn = self.rnn(x) 
                                   
        
        hn_last = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        
        hn_last = self.dropout(hn_last)
        output = self.fc(hn_last) 
        
        return output