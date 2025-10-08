import torch
from torch import nn

def LossRec(original_state: torch.Tensor, recon_state: torch.Tensor):
    criterion = nn.MSELoss()
    loss = criterion(recon_state, original_state)
    return loss

def LossPredls(original_state: torch.Tensor, recon_state: torch.Tensor, num_pred: float):
    criterion = nn.MSELoss()
    loss = criterion(recon_state, original_state) / num_pred
    return loss

def LossPredss(original_state: torch.Tensor, recon_state: torch.Tensor, num_pred: float):
    criterion = nn.MSELoss()
    loss = criterion(recon_state, original_state) / num_pred
    return loss
