import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.PatchEmbedding import PatchEmbedding
from layers.FourierCorrelation import FourierBlock
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.AutoCorrelation import AutoCorrelationLayer


# 引入上面定义的 Head
from layers.PatchEmbedding_K import QuantileFlattenHead

class Model(nn.Module):
    """
    Chapter 4: Probabilistic FEP-Former
    创新点：RevIN + Multi-Quantile Output
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ==================== 配置分位数 ====================
        # 建议配置: [0.1, 0.5, 0.9] 代表 10%, 50%(中位数), 90% 分位点
        self.quantiles = getattr(configs, 'quantiles', [0.1, 0.5, 0.9])
        self.num_quantiles = len(self.quantiles)
        print(f"ProbModel: Predicting quantiles {self.quantiles}")

        # ==================== 1. RevIN (创新点1) ====================
        self.revin = RevIN(
            num_features=configs.enc_in,
            affine=True,
            subtract_last=False
        )

        # ==================== Backbone (FEP-Former) ====================
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        self.patch_embedding = PatchEmbedding(
            seq_len=self.seq_len,
            patch_len=configs.patch_len,
            stride=configs.stride,
            d_model=self.d_model,
            dropout=configs.dropout
        )
        self.num_patch = self.patch_embedding.num_patch

        # ==================== 分解模块（保持FEDformer原有设计） ====================
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Encoder
        encoder_self_att = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.num_patch,
            modes=configs.modes,
            mode_select_method='random'
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, self.d_model, configs.n_heads),
                    self.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )

        # ==================== 2. 多分位数预测头 (创新点2) ====================
        # Seasonal Head: 输出 [pred_len * num_quantiles]
        self.head = QuantileFlattenHead(
            num_patch=self.num_patch,
            d_model=self.d_model,
            pred_len=self.pred_len,
            num_quantiles=self.num_quantiles,
            dropout=configs.dropout
        )

        # Trend Head: 仍然输出单点 [pred_len]，作为所有分位数的基准趋势
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [Batch, Seq_Len, N]

        # 1. RevIN Normalization
        x_enc = self.revin(x_enc, mode='norm')

        # 2. Decomposition
        seasonal_init, trend_init = self.decomp(x_enc)

        # 3. Trend Branch (Shared Trend)
        # [B, L, N] -> [B, N, L] -> [B, N, T] -> [B, T, N]
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        # 扩展维度以匹配分位数: [B, T, N] -> [B, T, N, 1]
        trend_out = trend_out.unsqueeze(-1)

        # 4. Seasonal Branch (Quantile Modeling)
        # Patching: [B, L, N] -> [B, N, M, D]
        enc_out = self.patch_embedding(seasonal_init)

        # CI: [B, N, M, D] -> [B*N, M, D]
        B, N, M, D = enc_out.shape
        enc_out = enc_out.reshape(B * N, M, D)

        # Encoder
        enc_out, attns = self.encoder(enc_out)

        # Quantile Head
        # Output shape: [B*N, T * Q]
        seasonal_out = self.head(enc_out)

        # Reshape to separate Quantiles
        # [B*N, T * Q] -> [B, N, T, Q]
        seasonal_out = seasonal_out.reshape(B, N, self.pred_len, self.num_quantiles)

        # Permute to [B, T, N, Q]
        seasonal_out = seasonal_out.permute(0, 2, 1, 3)

        # 5. Add Trend & Seasonal
        # Broadcasting: [B, T, N, Q] + [B, T, N, 1] = [B, T, N, Q]
        dec_out = seasonal_out + trend_out

        # 6. RevIN De-normalization (需要处理额外的 Q 维度)
        # 原始 RevIN 只能处理 [B, T, N]，我们需要手动广播 mean 和 std
        dec_out = self.denormalize_quantiles(dec_out)

        return dec_out

    def denormalize_quantiles(self, x):
        """
        处理带有分位数维度的反归一化
        x: [Batch, Pred_Len, N_Var, Quantiles]
        """
        # 获取 RevIN 存储的统计量: [Batch, 1, N_Var]
        mean = self.revin.mean
        stdev = self.revin.stdev

        # 扩展统计量维度: [Batch, 1, N, 1]
        mean = mean.unsqueeze(-1)
        stdev = stdev.unsqueeze(-1)

        # 仿射参数 (如果开启)
        if self.revin.affine:
            # weight/bias shape: [N] -> [1, 1, N, 1]
            weight = self.revin.affine_weight.view(1, 1, -1, 1)
            bias = self.revin.affine_bias.view(1, 1, -1, 1)
            x = x - bias
            x = x / (weight + self.revin.eps * self.revin.eps)

        # 标准反归一化
        x = x * stdev + mean
        return x