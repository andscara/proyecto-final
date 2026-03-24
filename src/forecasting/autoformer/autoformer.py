import torch
import torch.nn as nn

from forecasting.autoformer.auto_correlation import AutoCorrelation, AutoCorrelationLayer
from forecasting.autoformer.embed import DataEmbedding_with_exog
from forecasting.autoformer.encoder_decoder import Decoder, DecoderLayer, Encoder, EncoderLayer, MyLayernorm, SeriesDecomp


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(
            self,
            seq_len: int = 96,
            label_len: int = 48,
            pred_len: int = 24,
            output_attention: bool = False,
            moving_avg: int = 25,
            enc_in: int = 7,
            dec_in: int = 7,
            d_model: int = 512,
            embed: str = 'timeF',
            freq: str = 'h',
            dropout: float = 0.05,
            factor: int = 1,
            n_heads: int = 8,
            d_ff: int = 2048,
            activation: str = 'gelu',
            e_layers: int = 2,
            d_layers: int = 1,
            c_out: int = 7,
            d_mark: int | None = None,
            exog_c_in: int = 0,
            use_exog_vars: bool = True,
            use_temp_bins: bool = False,
            learnable_bins: bool = True,
    ):
        """
        Autoformer constructor
        :param seq_len: input sequence length
        :param label_len: start token length
        :param pred_len: prediction sequence length
        :param output_attention: whether to output attention in encoder
        :param moving_avg: window size of moving average
        :param enc_in: encoder input size
        :param dec_in: decoder input size
        :param d_model: dimension of model
        :param embed: time features encoding, options:[timeF, fixed, learned], default timeF
        :param freq: freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
        :param dropout: dropout rate
        :param factor: attn factor
        :param n_heads: num of heads
        :param d_ff: dimension of fcn
        :param activation: activation function of encoder and decoder
        :param e_layers: number of encoder layers
        :param d_layers: number of decoder layers
        :param c_out: output size
        """
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        self.decomp = SeriesDecomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_with_exog(enc_in, d_model, embed, freq,
                                                  dropout, d_mark=d_mark, exog_c_in=exog_c_in, use_exog_vars=use_exog_vars,
                                                  use_temp_bins=use_temp_bins, learnable_bins=learnable_bins)
        self.dec_embedding = DataEmbedding_with_exog(dec_in, d_model, embed, freq,
                                                  dropout, d_mark=d_mark, exog_c_in=exog_c_in, use_exog_vars=use_exog_vars,
                                                  use_temp_bins=use_temp_bins, learnable_bins=learnable_bins)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=MyLayernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=MyLayernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(
            self, 
            x_enc: torch.Tensor, 
            x_mark_enc: torch.Tensor, 
            x_dec: torch.Tensor, 
            x_mark_dec: torch.Tensor,
            enc_self_mask: torch.Tensor | None = None, 
            dec_self_mask: torch.Tensor | None = None, 
            dec_enc_mask: torch.Tensor | None = None
        ):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
