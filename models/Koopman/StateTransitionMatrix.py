import torch
from torch import nn

class StateTransitionMatrix(nn.Module):
    def __init__(self, latent_dim: int):
        super(StateTransitionMatrix, self).__init__()

        self.linear = nn.Linear(latent_dim, latent_dim)

    def forward(self, input: torch.Tensor):
        return self.linear(input)