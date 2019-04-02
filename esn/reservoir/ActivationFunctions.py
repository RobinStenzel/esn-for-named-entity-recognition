import torch


class HyperbolicTangent(object):
    def __call__(self, x):
        return torch.tanh(x)


class LogisticFunction(object):
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, x):
        return torch.sigmoid(self.beta * x)


class Linear(object):
    def __call__(self, x):
        return x


class SoftMax(object):
    def __call__(self, x):
        exp = torch.exp(x - torch.max(x))
        sum = torch.sum(exp)
        softmax = exp / sum
        return softmax