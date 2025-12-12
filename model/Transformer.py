import torch
from torch import nn
from .Attention import MaskedMultiHeadSelfAttention
from .FeedForward import FeedForward

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, expand_radio=4, dropout=0.1):
        super().__init__()
        self.attn = MaskedMultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, expand_radio, dropout=dropout)
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        """
        x.shape: [batch_size, seq_len, d_model]
        mask.shape: [batch_size, seq_len, seq_len]
        return: x.shape: [batch_size, seq_len, d_model]
        """
        out = self.ln_attn(x)
        out = self.attn(out, mask)
        x = out + x
        x = x + self.ff(self.ln_ff(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self, layers, vocab_size, d_model, num_heads, max_seq_len=2056, expand_radio=4, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout_embedding = nn.Dropout(dropout)
        self.transformers = nn.ModuleList([TransformerLayer(d_model, num_heads, expand_radio, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d_model)
        self.max_seq_len = max_seq_len

    def forward(self, x, content_lens):
        """
        x: [batch_size, seq_len]
        content_lens: (batch_size, )
        return: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        position = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.position_embedding(position)
        x = self.dropout_embedding(x)

        mask = torch.zeros((batch_size, seq_len, seq_len), device=x.device)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float("-inf"), diagonal=1)
        for i in range(batch_size):
            c = content_lens[i]
            mask[i, c:seq_len, c:seq_len] = causal_mask[c:seq_len, c:seq_len]

        for transformer in self.transformers:
            x = transformer(x, mask)

        x = self.ln(x)
        x = torch.matmul(x, self.token_embedding.weight.transpose(0, 1))
        return x


if __name__ == "__main__":
    x = torch.randint(0, 100, (16, 8), dtype=torch.int64)
    model = GPT2Model(3, 8724, 16, 4)
    output = model(x)
    print(output.shape)

