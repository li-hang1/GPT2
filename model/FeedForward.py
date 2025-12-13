from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model, expand_radio=4, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expand_radio)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_model * expand_radio, d_model)
    def forward(self, x):
        """
        x.shape: [batch_size, sequence_length, d_model]
        return: x.shape: [batch_size, sequence_length, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x