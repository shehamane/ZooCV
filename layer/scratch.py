import torch
from torch import nn


class BatchNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=1)
        vars = torch.var(x, dim=1)
        for c in range(self.num_features):
            x[:, c] = ((x[:, c] - means) / torch.sqrt(vars + self.eps)) * self.gamma[c] + self.beta[c]
        return x
