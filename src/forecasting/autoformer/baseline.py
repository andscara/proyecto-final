import torch
import torch.nn as nn


class Baseline(nn.Module):
    """
    Naive baseline: returns the last pred_len values of the encoder input,
    plus a learnable bias per horizon step so the model can be trained.
    """

    def __init__(self, seq_len: int = 336, pred_len: int = 168):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor
    ) -> torch.Tensor:
        out = x_enc[:, -self.pred_len:, 0]  # (B, pred_len)
        return out.unsqueeze(-1)  # (B, pred_len, 1)