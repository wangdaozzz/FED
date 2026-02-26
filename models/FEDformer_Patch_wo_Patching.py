import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.FourierCorrelation import FourierBlock
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.Embed import DataEmbedding_wo_pos  # 回退到使用原生的Embedding


class Model(nn.Module):
    """
    Ablation Study: w/o Patching

    特征:
    1. 移除 PatchEmbedding，使用点对点的 DataEmbedding_wo_pos。
    2. 保留 Channel Independence (CI) 策略。
    3. 保留 Encoder-Only + FlattenHead 结构。
    4. FourierBlock 作用于原始 seq_len 长度。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model

        # ==================== Embedding (改回 Point-wise) ====================
        # 因为保留了 Channel Independence，每个变量独立进入，所以 c_in = 1
        self.enc_embedding = DataEmbedding_wo_pos(
            c_in=1,
            d_model=configs.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout
        )

        # ==================== 分解模块 ====================
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # ==================== 频域 Attention ====================
        # 关键修改：seq_len 变回原始长度，不再是 num_patch
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 32)

        # 这里的 seq_len 是原始序列长度 (如 96)
        # 相比 Patch 版本，这里的计算量会增加 (O(L) vs O(L/S))
        enc_modes = int(min(self.modes, self.seq_len // 2))
        print(f'w/o Patching Mode: seq_len={self.seq_len}, enc_modes={enc_modes}')

        encoder_self_att = FourierBlock(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            seq_len=self.seq_len,  # 作用于全长
            modes=enc_modes,
            mode_select_method=self.mode_select
        )

        # ==================== Encoder ====================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # ==================== 输出投影层 ====================
        # 输入维度是 seq_len * d_model
        self.head = nn.Linear(self.seq_len * configs.d_model, self.pred_len)
        self.dropout = nn.Dropout(configs.dropout)

        # Trend 部分
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # x_enc: [Batch, Seq_len, Num_variables]
        B, L, N = x_enc.shape

        # 1. 分解
        seasonal_init, trend_init = self.decomp(x_enc)

        # 2. Trend 投影 (DLinear 方式)
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 3. Channel Independence 处理
        # 将多变量拆解为多个单变量序列：[B, L, N] -> [B*N, L, 1]
        seasonal_init = seasonal_init.permute(0, 2, 1).reshape(B * N, L, 1)

        # CI 模式下，x_mark_enc 也需要扩展匹配 (如果 Embedding 用到了的话)
        # 这里的 DataEmbedding_wo_pos 内部会处理 temporal embedding，需要对应的 mark
        # [B, L, C_time] -> [B, 1, L, C_time] -> [B, N, L, C_time] -> [B*N, L, C_time]
        if x_mark_enc is not None:
            x_mark_enc = x_mark_enc.unsqueeze(1).repeat(1, N, 1, 1).reshape(B * N, L, -1)

        # 4. Embedding (Point-wise)
        # [B*N, L, 1] -> [B*N, L, d_model]
        enc_out = self.enc_embedding(seasonal_init, x_mark_enc)

        # 5. Encoder (频域 Attention)
        # [B*N, L, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 6. Flatten Head
        # [B*N, L, d_model] -> [B*N, L * d_model]
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        # -> [B*N, pred_len]
        seasonal_out = self.dropout(self.head(enc_out))

        # 7. Reshape 回多变量形式
        # [B*N, pred_len] -> [B, N, pred_len] -> [B, pred_len, N]
        seasonal_out = seasonal_out.reshape(B, N, self.pred_len).permute(0, 2, 1)

        # 8. Add Trend
        dec_out = seasonal_out + trend_out

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out