import torch
from torch import nn

class ActionEncoder(nn.Module):
    '''
    エージェントのアクションを潜在空間にマッピングする
    ・action_dim : エージェントのアクションの次元
    ・latent_dim : マッピングする潜在空間の次元
    '''
    def __init__(self, action_dim: int, latent_dim: int, hidden_dim=128):
        super(ActionEncoder, self).__init__()

        #*--- モデルの層の構造 ---*#
        self.linear1 = nn.Linear(action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input: torch.Tensor):
        x = self.linear1(input)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        x = self.linear3(x)

        return x
