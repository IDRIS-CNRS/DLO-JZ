class Duplication(Function):
    @staticmethod
    def forward(ctx, x):
        ###### BALISE 1a ######
        return x
        ###### FIN BALISE 1a ######

    @staticmethod
    def backward(ctx, grad_output):
        ###### BALISE 1b / BALISE 9a ######
        dist.all_reduce(grad_output)
        return grad_output
        ###### FIN BALISE 1b / BALISE 9a ######


class AllReduce(Function):
    @staticmethod
    def forward(ctx, x):
        ###### BALISE 1c / BALISE 9b ######
        dist.all_reduce(x)
        return x
        ###### FIN BALISE 1c / BALISE 9b ######

    @staticmethod
    def backward(ctx, grad_output):
        ###### BALISE 1d ######
        return grad_output
        ###### FIN BALISE 1d ######
