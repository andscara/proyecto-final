#type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Temperature thresholds in normalized scale: temp_norm = (celsius + 15) / 65
# muy_frio    : celsius < 5            → norm < 0.308
# frio        : 5  ≤ celsius < 12      → 0.308 ≤ norm < 0.415
# calor       : 28 ≤ celsius < 35      → 0.662 ≤ norm < 0.769
# mucho_calor : celsius ≥ 35           → norm ≥ 0.769
# normal (12–28 °C): all four = 0
_TEMP_MUY_FRIO    = (5  + 15) / 65   # 0.308
_TEMP_FRIO        = (12 + 15) / 65   # 0.415
_TEMP_CALOR       = (28 + 15) / 65   # 0.662
_TEMP_MUCHO_CALOR = (35 + 15) / 65   # 0.769
NUM_TEMP_BINS = 4


def temp_bins(t: torch.Tensor) -> torch.Tensor:
    """
    t: (B, T) normalized temperature values.
    Returns (B, T, 4) float binary indicators: [muy_frio, frio, calor, mucho_calor].
    """
    muy_frio    = (t < _TEMP_MUY_FRIO).float()
    frio        = ((t >= _TEMP_MUY_FRIO) & (t < _TEMP_FRIO)).float()
    calor       = ((t >= _TEMP_CALOR)    & (t < _TEMP_MUCHO_CALOR)).float()
    mucho_calor = (t >= _TEMP_MUCHO_CALOR).float()
    return torch.stack([muy_frio, frio, calor, mucho_calor], dim=-1)  # (B, T, 4)


class TempEncoder(nn.Module):
    """
    Learnable soft temperature bins with optional temporal context.

    Uses sigmoid step-functions at learnable thresholds (initialized at the
    original hand-crafted values).  When use_context=True the thresholds are
    shifted by a learned function of month, day of month, hour, and holidays:

        θ_i(month, day, hour, holiday) = θ_i_base  +  W_i · [month_norm, day_norm, hour_norm, is_holiday]

    This lets the model learn, e.g., that at night the "cold" threshold is
    lower (people are asleep, less temperature-driven demand), that in
    summer the "hot" threshold shifts, that at the end of the month
    people use less heating/cooling, or that on holidays people react
    differently to temperature.  Weights are initialised at **zero**
    so the model starts from the exact same solution as the fixed bins.
    """

    def __init__(self, n_bins: int = NUM_TEMP_BINS, use_context: bool = False):
        super().__init__()
        self.n_bins = n_bins
        self.use_context = use_context
        # Initialize thresholds at the original hand-crafted boundaries
        init_thresholds = torch.tensor(
            [_TEMP_MUY_FRIO, _TEMP_FRIO, _TEMP_CALOR, _TEMP_MUCHO_CALOR]
        )
        if n_bins != len(init_thresholds):
            init_thresholds = torch.linspace(
                _TEMP_MUY_FRIO, _TEMP_MUCHO_CALOR, n_bins
            )
        self.thresholds = nn.Parameter(init_thresholds)
        # Sharpness controls transition steepness (higher → closer to hard bins)
        self.log_sharpness = nn.Parameter(torch.full((n_bins,), math.log(20.0)))
        if use_context:
            # [month_norm, day_norm, hour_norm, is_holiday] → per-bin threshold offset
            self.context_proj = nn.Linear(4, n_bins, bias=False)
            nn.init.zeros_(self.context_proj.weight)

    def forward(
        self,
        t: torch.Tensor,
        month: torch.Tensor | None = None,
        day: torch.Tensor | None = None,
        hour: torch.Tensor | None = None,
        is_holiday: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        t:          (B, T)  normalized temperature values.
        month:      (B, T)  raw month 1-12 (optional, required if use_context).
        day:        (B, T)  raw day 1-31   (optional, required if use_context).
        hour:       (B, T)  raw hour 0-23  (optional, required if use_context).
        is_holiday: (B, T)  binary 0/1     (optional, required if use_context).
        Returns (B, T, n_bins) soft step features.
        """
        thresholds = self.thresholds                                # (n_bins,)
        if self.use_context and month is not None and day is not None and hour is not None and is_holiday is not None:
            month_norm = (month - 1.0) / 11.0 - 0.5                # → [-0.5, 0.5]
            day_norm   = (day - 1.0) / 30.0 - 0.5                  # → [-0.5, 0.5]
            hour_norm  = hour / 23.0 - 0.5                         # → [-0.5, 0.5]
            ctx = torch.stack([month_norm, day_norm, hour_norm, is_holiday], dim=-1)  # (B, T, 4)
            thresholds = thresholds + self.context_proj(ctx)        # (B, T, n_bins)
        sharpness = self.log_sharpness.exp()
        return torch.sigmoid(sharpness * (t.unsqueeze(-1) - thresholds))

def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")
    
    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        if x.device.type == 'mps':
            # padding_mode='circular' can produce wrong gradients on MPS; compute on CPU
            x_perm = x.permute(0, 2, 1).contiguous().cpu()  # [B, C_in, L]
            w = self.tokenConv.weight.cpu()
            # Manual circular padding: pad left with last element, right with first
            padded = torch.cat([x_perm[:, :, -1:], x_perm, x_perm[:, :, :1]], dim=-1)
            return F.conv1d(padded, w).transpose(1, 2).to(x.device)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h', use_holidays: bool = True):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        self.holiday_embed = nn.Embedding(2, d_model) # binary holiday feature
        self.use_holidays = use_holidays

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        x = hour_x + weekday_x + day_x + month_x + minute_x
        if self.use_holidays:
            x = x + self.holiday_embed(x[:, :, 5])
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h', d_mark: int | None = None):
        super(TimeFeatureEmbedding, self).__init__()

        if d_mark is not None:
            d_inp = d_mark
        else:
            freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
            d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, d_mark: int | None = None):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq, d_mark=d_mark)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
    
class DataEmbedding_with_exog(nn.Module):
    def __init__(
            self,
            c_in,
            d_model,
            embed_type='fixed',
            freq='h',
            dropout=0.1,
            d_mark: int | None = None,
            exog_c_in: int = 0,
            use_exog_vars: bool = True,
        ):
        super(DataEmbedding_with_exog, self).__init__()
        self.d_mark = d_mark
        self.exog_c_in = exog_c_in
        self.use_exog_vars = use_exog_vars
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.exog_embedding = nn.Linear(in_features=exog_c_in, out_features=d_model, bias=False)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq, use_holidays=use_exog_vars) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq, d_mark=d_mark)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark[:, :, :self.d_mark])
        if self.use_exog_vars and self.exog_c_in > 0:
            x = x + self.exog_embedding(x_mark[:, :, -self.exog_c_in:])
        return self.dropout(x)
