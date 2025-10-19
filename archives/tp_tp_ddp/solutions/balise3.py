class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        ###### BALISE 3a ######
        hidden_dim = hidden_dim // pcomm.tp_degree
        self.mod = nn.Embedding(vocab_size, hidden_dim)
        ###### FIN BALISE 3a ######

    def forward(self, x):
        ###### BALISE 3b ######
        x = self.mod(x)
        x = pcomm.AllGather.apply(x)
        ###### FIN BALISE 3b ######
        return x