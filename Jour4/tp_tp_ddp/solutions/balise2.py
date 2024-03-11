class ColWiseLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        ###### BALISE 2a ######
        out_dim = out_dim // pcomm.tp_degree
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        ###### FIN BALISE 2a ######

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ###### BALISE 2b ######
        x = pcomm.Duplication.apply(x)
        x = self.linear(x)
        x = pcomm.AllGather.apply(x)
        ###### FIN BALISE 2b ######
        return x