import torch
import torch.nn as nn


class LinearBaseline2(nn.Module):
    """
    Simple linear baseline: flattens the encoder input (seq_len,)
    and projects it to (pred_len,) with a single linear layer.
    """

    def __init__(self, seq_len: int = 336, pred_len: int = 168):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor
    ) -> torch.Tensor:
        out = x_enc[:, -self.pred_len:, 0]  # (B, pred_len)
        return out.unsqueeze(-1)  # (B, pred_len, 1)