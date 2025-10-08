import torch
from torch import nn
import torch.nn.init as init

def orthogonal_init(layer, gain="tanh", value = 0.01):
    if gain == "custom":
        gain = value
    else:
        gain = init.calculate_gain(gain)
    init.orthogonal_(layer.weight, gain)

class MLPCritic(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256):
        super(MLPCritic, self).__init__()

        #*--- V model ---*#
        self.Linear1 = nn.Linear(latent_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.Value = nn.Linear(hidden_dim, 1)

        #*--- Orthogonal initialization and layer scaling ---*#
        orthogonal_init(self.Linear1)
        orthogonal_init(self.Linear2)
        orthogonal_init(self.Value, "custom", value=1.0)

    def forward(self, latent_state):
        
        out = self.Linear1(latent_state)
        out = torch.tanh(out)
        out = self.Linear2(out)
        out = torch.tanh(out)
        v = self.Value(out)

        return v