class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_prob: float) -> None:
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        ###### BALISE 4a ######
        self.num_heads = num_heads // pcomm.tp_degree
        self.mid_dim = self.head_dim * self.num_heads
        ###### FIN BALISE 4a ######

        self.output = nn.Linear(self.mid_dim, hidden_dim, bias=False)
        self.q = nn.Linear(hidden_dim, self.mid_dim, bias=False)
        self.k = nn.Linear(hidden_dim, self.mid_dim, bias=False)
        self.v = nn.Linear(hidden_dim, self.mid_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size()[:-2] + (self.mid_dim,))
        return x

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ###### BALISE 4b ######
        x = pcomm.Duplication.apply(x)
        q = self.separate_heads(self.q(x))
        k = self.separate_heads(self.k(x))
        v = self.separate_heads(self.v(x))
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_dim) + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        y = torch.matmul(attention_probs, v)
        y = self.merge_heads(y)

        output = self.output(y)
        output = pcomm.AllReduce.apply(output)
        ###### FIN BALISE 4b ######
        return output