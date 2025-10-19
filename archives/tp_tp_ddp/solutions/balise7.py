class Block(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int, dropout_prob: float) -> None:
        super().__init__()
        self.attention = ResNorm(MultiHeadSelfAttention(hidden_dim, num_heads, dropout_prob), hidden_dim, dropout_prob)
        ###### BALISE 7a ######
        self.ff = ResNorm(
            ColRowLinearPair(hidden_dim, intermediate_dim, hidden_dim, [nn.GELU()]),
            hidden_dim, dropout_prob,
        )
        ###### FIN BALISE 7a ######

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.attention(x, attention_mask=attention_mask)
        x = self.ff(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_dim: int, dropout_prob: float) -> None:
        super().__init__()
        ###### BALISE 7b ######
        self.network = ColRowLinearPair(
            in_dim=hidden_dim,
            mid_dim=hidden_dim,
            out_dim=2,
            pointwise_mid_modules=[nn.GELU(), nn.Dropout(dropout_prob)],
        )
        ###### FIN BALISE 7b ######

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, :] # only take the output of the [CLS] token
        x = self.network(x)
        return x