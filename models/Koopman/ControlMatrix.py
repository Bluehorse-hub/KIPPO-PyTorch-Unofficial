import torch
from torch import nn

class ControlMatrix(nn.Module):
    '''
    制御行列B
    潜在空間にマッピングされたアクションに対して作用する制御行列
    Koopman理論は線形変換に帰着させる理論のためここでは活性化関数を用いた非線形化は行わない
    '''
    def __init__(self, latent_dim: int):
        super(ControlMatrix, self).__init__()

        self.linear = nn.Linear(latent_dim, latent_dim)

    def forward(self, input: torch.Tensor):
        return self.linear(input)