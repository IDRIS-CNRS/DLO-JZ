class Duplication(Function):
    @staticmethod
    def forward(ctx, x):
        ###### BALISE 1a ######
        return x
        ###### FIN BALISE 1a ######

    @staticmethod
    def backward(ctx, grad_output):
        ###### BALISE 1b / BALISE 9a ######
        dist.all_reduce(grad_output, group=tp_grp)
        return grad_output
        ###### FIN BALISE 1b / BALISE 9a ######


class AllReduce(Function):
    @staticmethod
    def forward(ctx, x):
        ###### BALISE 1c / BALISE 9b ######
        dist.all_reduce(x, group=tp_grp)
        return x
        ###### FIN BALISE 1c / BALISE 9b ######

    @staticmethod
    def backward(ctx, grad_output):
        ###### BALISE 1d ######
        return grad_output
        ###### FIN BALISE 1d ######


class AllGather(Function):
    @staticmethod
    def forward(ctx, x):
        out_list = [torch.zeros_like(x) for _ in range(tp_degree)]
        ###### BALISE 9c ######
        dist.all_gather(out_list, x, group=tp_grp)
        ###### FIN BALISE 9c ######
        out = torch.cat(out_list, dim=-1)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        dim = grad_outputs.size()[-1] // tp_degree
        return grad_outputs[..., dim * tp_rank : dim * (tp_rank + 1)]