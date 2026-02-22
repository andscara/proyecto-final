import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLayernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels: int):
        super(MyLayernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == 'mps':
            # LayerNorm on MPS can produce near-zero gradients for weight/bias,
            # causing Adam to blow up when a real gradient finally arrives. Run on CPU.
            x_cpu = x.cpu()
            w = self.layernorm.weight.cpu()
            b = self.layernorm.bias.cpu()
            x_hat = F.layer_norm(x_cpu, self.layernorm.normalized_shape, w, b, self.layernorm.eps)
            x_hat = x_hat.to(x.device)
        else:
            x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(
            self, 
            kernel_size: int,
            stride: int
        ):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(
            self, 
            attention: nn.Module, 
            d_model: int, 
            d_ff: int | None = None, 
            moving_avg: int = 25, 
            dropout: float = 0.1, 
            activation: str = "relu"
        ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(
            self,
            attn_layers: list[nn.Module],
            conv_layers: list[nn.Module] | None = None,
            norm_layer: nn.Module | None = None
        ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(
            self, 
            self_attention: nn.Module, 
            cross_attention: nn.Module, 
            d_model: int, 
            c_out: int, 
            d_ff: int | None = None,
            moving_avg: int = 25, 
            dropout: float = 0.1, 
            activation: str = "relu"
        ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='replicate', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: torch.Tensor | None = None, cross_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        if residual_trend.device.type == 'mps':
            # padding_mode='replicate' can produce wrong gradients on MPS; compute on CPU
            rt_cpu = residual_trend.permute(0, 2, 1).contiguous().cpu()  # [B, d_model, L]
            w = self.projection.weight.cpu()
            # Manual replicate padding: pad left with first element, right with last
            padded = torch.cat([rt_cpu[:, :, :1], rt_cpu, rt_cpu[:, :, -1:]], dim=-1)
            residual_trend = F.conv1d(padded, w).transpose(1, 2).to(trend1.device)
        else:
            residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(
            self, 
            layers: list[nn.Module], 
            norm_layer: nn.Module | None = None, 
            projection: nn.Module | None = None
        ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: torch.Tensor | None = None, cross_mask: torch.Tensor | None = None, trend: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend # type: ignore
