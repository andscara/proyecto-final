import torch
import torch.nn as nn

class SMAPE(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        numerator = torch.abs(y_pred - y_true)
        denominator = torch.abs(y_true) + torch.abs(y_pred) + self.eps
        return torch.mean(2.0 * numerator / denominator)
