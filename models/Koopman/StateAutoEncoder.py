import torch
from torch import nn

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim=128):
        super(StateEncoder, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input: torch.Tensor):
        x = self.linear1(input)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        x = self.linear3(x)

        return x

class StateDecoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim=128):
        super(StateDecoder, self).__init__()

        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, input: torch.Tensor):
        y = self.linear1(input)
        y = torch.tanh(y)
        y = self.linear2(y)
        y = torch.tanh(y)
        y = self.linear3(y)

        return y