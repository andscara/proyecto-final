import torch
import torch.nn as nn

from forecasting.autoformer.embed import NUM_TEMP_BINS, temp_bins

HOLIDAY_MARK_IDX = 4  # position of is_holiday in x_mark_enc (month, day, weekday, hour, is_holiday, ...)


class LinearBaseline(nn.Module):
    """
    Simple linear baseline: flattens the encoder input (seq_len,)
    and projects it to (pred_len,) with a single linear layer.

    include_holiday: if True, includes:
        - past is_holiday from x_mark_enc (index 4), shape (seq_len,)
        - future is_holiday from x_mark_dec (index 4), shape (pred_len,)
        The model thus knows which days in the prediction horizon are holidays.
    exog_size: number of columns taken from the END of x_mark_enc/dec (e.g. temp_media).
        Past (seq_len) and future (pred_len) exog are both included linearly.
    temp_bins: if True, assumes the first (or only) exog column is temp_media and appends
        4 binary indicators (muy_frio, frio, calor, mucho_calor) per timestep,
        allowing the linear model to capture the V-shaped temperature response.
    """

    def __init__(
        self,
        seq_len: int = 336,
        pred_len: int = 168,
        exog_size: int = 0,
        include_holiday: bool = False,
        use_temp_bins: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.exog_size = exog_size
        self.include_holiday = include_holiday
        self.use_temp_bins = use_temp_bins and exog_size > 0

        holiday_size = (seq_len + pred_len) if include_holiday else 0
        bins_size = (seq_len + pred_len) * NUM_TEMP_BINS if self.use_temp_bins else 0
        exog_flat_size = (seq_len + pred_len) * exog_size

        total_input = seq_len + holiday_size + exog_flat_size + bins_size
        print(f"[LinearBaseline] use_temp_bins={self.use_temp_bins}, exog_size={exog_size}, input_size={total_input}")
        self.linear = nn.Linear(total_input, pred_len)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor
    ) -> torch.Tensor:
        inp = x_enc[:, :, 0]  # (B, seq_len)
        if self.include_holiday:
            past_holiday = x_mark_enc[:, :, HOLIDAY_MARK_IDX]                      # (B, seq_len)
            future_holiday = x_mark_dec[:, -self.pred_len:, HOLIDAY_MARK_IDX]      # (B, pred_len)
            inp = torch.cat([inp, past_holiday, future_holiday], dim=-1)
        if self.exog_size > 0:
            past_exog = x_mark_enc[:, :, -self.exog_size:]                         # (B, seq_len, exog_size)
            future_exog = x_mark_dec[:, -self.pred_len:, -self.exog_size:]         # (B, pred_len, exog_size)
            inp = torch.cat([inp, past_exog.flatten(1), future_exog.flatten(1)], dim=-1)
            if self.use_temp_bins:
                past_bins = temp_bins(past_exog[:, :, 0]).flatten(1)               # (B, seq_len * 4)
                future_bins = temp_bins(future_exog[:, :, 0]).flatten(1)           # (B, pred_len * 4)
                inp = torch.cat([inp, past_bins, future_bins], dim=-1)
        out = self.linear(inp)  # (B, pred_len)
        return out.unsqueeze(-1)  # (B, pred_len, 1)
