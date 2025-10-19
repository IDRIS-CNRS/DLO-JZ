class ColRowLinearPair(nn.Module):
    def __init__(
        self, in_dim: int, mid_dim: int, out_dim: int, pointwise_mid_modules: list[nn.Module]
    ) -> None:
        super().__init__()
        
        ###### BALISE 6a ######
        mid_dim = mid_dim // pcomm.tp_degree
        self.first_linear = nn.Linear(in_dim, mid_dim, bias=False)
        self.second_linear = nn.Linear(mid_dim, out_dim, bias=False)
        ###### FIN BALISE 6a ######
        self.mid = nn.ModuleList(pointwise_mid_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ###### BALISE 6b ######
        x = pcomm.Duplication.apply(x)
        x = self.first_linear(x)
        for layer in self.mid:
            x = layer(x)
        x = self.second_linear(x)
        x = pcomm.AllReduce.apply(x)
        ###### FIN BALISE 6b ######
        return x