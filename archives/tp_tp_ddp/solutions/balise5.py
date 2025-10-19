class FeedForwardBlock(nn.Module):
    def __init__(
        self, in_dim: int, mid_dim: int, out_dim: int, pointwise_mid_modules: list[nn.Module],
    ) -> None:
        super().__init__()
        ###### BALISE 5 ######
        self.first_layer = ColWiseLinear(in_dim, mid_dim)
        self.mid = nn.ModuleList(pointwise_mid_modules)
        self.second_layer = ColWiseLinear(mid_dim, out_dim)
        ###### FIN BALISE 5 ######

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        for layer in self.mid:
            x = layer(x)
        x = self.second_layer(x)
        return x