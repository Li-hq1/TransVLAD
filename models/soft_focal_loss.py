from torch import nn
import torch
from torch.nn import functional as F


class soft_focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(soft_focal_loss, self).__init__()
        self.gamma = gamma

    def forward(self, x, targets):
        log_preds = F.log_softmax(x, dim=-1)
        preds = torch.exp(log_preds)
        loss = -torch.pow((targets - preds) ** 2, self.gamma / 2) * targets * log_preds
        loss = torch.sum(loss, dim=-1)
        return loss.mean()
