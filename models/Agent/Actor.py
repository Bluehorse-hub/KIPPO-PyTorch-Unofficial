import torch
from torch import nn
import torch.nn.init as init
from torch.distributions import Normal

def orthogonal_init(layer, gain="tanh", value = 0.01):
    if gain == "custom":
        gain = value
    else:
        gain = init.calculate_gain(gain)
    init.orthogonal_(layer.weight, gain)

class MLPPolicy(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super(MLPPolicy, self).__init__()

        #*--- モデルの層の構造 ---*#
        self.Linear1 = nn.Linear(latent_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear_mean = nn.Linear(hidden_dim, action_dim)

        #*--- Orthogonal initialization and layer scaling ---*#
        orthogonal_init(self.Linear1)
        orthogonal_init(self.Linear2)
        orthogonal_init(self.Linear_mean, "custom", value=0.01)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, latent_state):
        out = self.Linear1(latent_state)
        out = torch.tanh(out)
        out = self.Linear2(out)
        out = torch.tanh(out)

        mean = self.Linear_mean(out)
        std = torch.exp(self.log_std).expand_as(mean)

        if torch.isnan(mean).any():
            print("NaN in mean! state min:", latent_state.min().item(), "max:", latent_state.max().item())
        
        return mean, std
    
    def sample_action(self, latent_state, test=False):
        policy = self.policy(latent_state)
        if test == True:
            action = policy.mean
            log_prob = None
        else:
            action = policy.sample()
            log_prob = policy.log_prob(action).sum(dim=-1)
        action_tanh = torch.tanh(action)
        return action, log_prob, action_tanh
    
    def policy(self, latent_state):
        means, _ = self.forward(latent_state)
        std = torch.exp(self.log_std).expand_as(means)
        std = torch.clamp(std, 1e-3, 1.0)
        policy = Normal(means, std)
        return policy