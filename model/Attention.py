import torch
from torch import nn
import torch.nn.functional as F

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        """
        x.shape: (batch_size, sequence_length, d_model)
        mask.shape: (batch_size, sequence_length, sequence_length)
        return: output.shape: (batch_size, sequence_length, d_model)
        """
        Q, K, V = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        weight = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.d_k ** 0.5 + mask.unsqueeze(1), dim=-1)
        weight = self.dropout(weight)
        output = self.linear(torch.matmul(weight, V).transpose(1, 2).reshape(weight.shape[0], -1, self.d_model))
        return output

if __name__ == "__main__":
    model = MaskedMultiHeadSelfAttention(d_model=16, num_heads=2)
    x = torch.randn(32, 10, 16)
    mask = torch.randn(10, 10)
    y = model(x, mask)
    print(y.shape)


