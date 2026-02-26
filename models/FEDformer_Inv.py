import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入FEDformer组件
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.Autoformer_EncDec import (
    Encoder, EncoderLayer, my_Layernorm,
    series_decomp, series_decomp_multi
)
from layers.AutoCorrelation import AutoCorrelationLayer

# 导入Inverted组件
from layers.InvertedEmbedding import InvertedEmbedding, InvertedOutputHead


class Model(nn.Module):
    """
    FEDformer + Inverted

    核心创新：
    1. 频域Attention (FEDformer原有): 使用Fourier变换在频域做attention
    2. Inverted (新增): 将变量作为token，在变量维度做attention

    架构改变：
    - 原始FEDformer: [B, L, N] -> 时间维度attention -> [B, pred_len, N]
    - FEDformer_Inv: [B, L, N] -> 变量维度attention -> [B, pred_len, N]

    核心思想：
    - 传统方法在时间维度建模，每个时间点是一个token
    - Inverted在变量维度建模，每个变量是一个token，能捕获变量间的相关性
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ==================== 基本参数 ====================
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.output_attention = configs.output_attention

        # ==================== 分解模块 ====================
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            kernel_size = kernel_size[0]
        self.decomp = series_decomp(kernel_size)

        # ==================== Fourier参数 ====================
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 64)

        print(f'[FEDformer_Inv] 配置信息:')
        print(f'  - seq_len={self.seq_len}, pred_len={self.pred_len}')
        print(f'  - enc_in (变量数)={self.enc_in}')
        print(f'  - d_model={self.d_model}, n_heads={self.n_heads}, e_layers={self.e_layers}')

        # ==================== Inverted Embedding ====================
        # 将每个变量的整个时间序列映射到d_model
        self.enc_embedding = InvertedEmbedding(
            seq_len=self.seq_len,
            d_model=self.d_model,
            dropout=self.dropout
        )

        # ==================== Encoder (FourierBlock在变量维度) ====================
        # FourierBlock的seq_len参数设为变量数，在变量维度做频域attention
        variate_modes = int(min(self.modes, self.enc_in // 2))
        variate_modes = max(variate_modes, 1)  # 至少1个mode

        print(f'  - Variate FourierBlock: seq_len={self.enc_in}, modes={variate_modes}')

        variate_fourier = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.enc_in,  # 变量数作为序列长度
            modes=variate_modes,
            mode_select_method=self.mode_select
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        variate_fourier,
                        self.d_model,
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    moving_avg=kernel_size,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )

        # ==================== 输出投影 ====================
        # Seasonal部分: Inverted输出头
        self.head = InvertedOutputHead(
            d_model=self.d_model,
            pred_len=self.pred_len,
            dropout=self.dropout
        )

        # Trend部分: 简单线性投影
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        x_enc: [Batch, Seq_len, Num_variables]

        return: [Batch, Pred_len, Num_variables]
        """
        B, L, N = x_enc.shape

        # ==================== Step 1: 序列分解 ====================
        seasonal_init, trend_init = self.decomp(x_enc)

        # ==================== Step 2: Trend投影 ====================
        # [B, L, N] -> [B, N, L] -> [B, N, pred_len] -> [B, pred_len, N]
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1))
        trend_out = trend_out.permute(0, 2, 1)

        # ==================== Step 3: Inverted Embedding ====================
        # [B, L, N] -> [B, N, d_model]
        # 每个变量的整个时间序列被映射到d_model维度
        enc_out = self.enc_embedding(seasonal_init)

        # ==================== Step 4: Encoder (变量维度FourierBlock) ====================
        # 在变量维度做频域attention，捕获变量间的相关性
        # enc_out: [B, N, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # ==================== Step 5: 输出投影 ====================
        # [B, N, d_model] -> [B, pred_len, N]
        seasonal_out = self.head(enc_out)

        # ==================== Step 6: 合并趋势和季节 ====================
        dec_out = seasonal_out + trend_out

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out


if __name__ == '__main__':
    """测试代码"""


    class Configs:
        seq_len = 96
        label_len = 48
        pred_len = 96
        enc_in = 8
        dec_in = 8
        c_out = 8
        d_model = 512
        n_heads = 8
        d_ff = 2048
        e_layers = 2
        d_layers = 1
        dropout = 0.1
        activation = 'gelu'
        moving_avg = 25
        output_attention = False
        mode_select = 'random'
        modes = 64


    configs = Configs()
    model = Model(configs)

    print('=' * 60)
    print('模型参数量: {:,}'.format(sum(p.numel() for p in model.parameters())))
    print('=' * 60)

    # 测试前向传播
    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)

    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print(f'\n输入shape: {x_enc.shape}')
    print(f'输出shape: {out.shape}')
    print(f'期望shape: [{batch_size}, {configs.pred_len}, {configs.c_out}]')

    assert out.shape == (batch_size, configs.pred_len, configs.c_out), "Shape不匹配!"
    print('\n✓ 模型测试通过!')