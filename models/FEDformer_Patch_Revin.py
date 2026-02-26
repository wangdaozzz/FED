import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 导入原有的FEDformer组件
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.RevIN import RevIN
# 导入新建的Patch组件
from layers.PatchEmbedding import PatchEmbedding, FlattenHead


class Model(nn.Module):
    """
    FEDformer + Patching

    核心改动：
    1. 用PatchEmbedding替换原有的DataEmbedding
    2. FourierBlock在patch维度上做频域attention
    3. 采用Channel Independence策略：每个变量独立处理
    4. 简化为Encoder-only结构
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ==================== 基本参数 ====================
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 输入变量数
        self.c_out = configs.c_out  # 输出变量数
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention

        # ==================== Patch参数 ====================
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)

        # ==================== Patch Embedding（新增） ====================
        # 注意：num_patch由PatchEmbedding内部计算
        self.patch_embedding = PatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            dropout=configs.dropout
        )
        # RevIn
        self.revin = True
        # if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.revin_layer = RevIN(
            num_features=configs.enc_in,  # 输入变量数
            affine=getattr(configs, 'affine', True),  # 是否使用可学习参数
            subtract_last=getattr(configs, 'subtract_last', False)  # 是否使用Last-value模式
        )

        # 从patch_embedding获取实际的num_patch
        self.num_patch = self.patch_embedding.num_patch

        print(f'模型配置: patch_len={self.patch_len}, stride={self.stride}, num_patch={self.num_patch}')

        # ==================== 分解模块（保持FEDformer原有设计） ====================
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # ==================== 频域Attention配置 ====================
        # FourierBlock的seq_len改为num_patch（关键修改！）
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 32)
        enc_modes = int(min(self.modes, self.num_patch // 2))
        print(f'Fourier modes: {enc_modes}')

        encoder_self_att = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.num_patch,  # 关键：从seq_len改为num_patch
            modes=enc_modes,
            mode_select_method=self.mode_select
        )

        # ==================== Encoder ====================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model,
                        configs.n_heads
                    ),
                    self.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )

        # ==================== 输出投影层 ====================
        # Seasonal部分: Patch展平后投影
        self.head = FlattenHead(
            num_patch=self.num_patch,
            d_model=self.d_model,
            pred_len=self.pred_len,
            dropout=configs.dropout
        )

        # Trend部分: 简单线性投影
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        x_enc: [Batch, Seq_len, Num_variables]
        x_mark_enc: [Batch, Seq_len, Time_features] (本模型中不使用)

        return: [Batch, Pred_len, Num_variables]
        """
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')

        B, L, N = x_enc.shape

        # ==================== Step 1: 序列分解 ====================
        # seasonal_init: [B, L, N], trend_init: [B, L, N]
        seasonal_init, trend_init = self.decomp(x_enc)

        # ==================== Step 2: Trend部分处理 ====================
        # 简单线性投影趋势项
        # [B, L, N] -> [B, N, L] -> [B, N, pred_len] -> [B, pred_len, N]
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1))
        trend_out = trend_out.permute(0, 2, 1)  # [B, pred_len, N]

        # ==================== Step 3: Seasonal部分 - Patch Embedding ====================
        # [B, L, N] -> [B, N, num_patch, d_model]
        enc_out = self.patch_embedding(seasonal_init)

        # ==================== Step 4: Channel Independence ====================
        # 将每个变量独立处理，共享同一个Encoder
        # [B, N, num_patch, d_model] -> [B*N, num_patch, d_model]
        enc_out = enc_out.reshape(B * N, self.num_patch, self.d_model)

        # ==================== Step 5: Encoder (频域Attention) ====================
        # FourierBlock在patch维度做频域attention
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # ==================== Step 6: 恢复变量维度 ====================
        # [B*N, num_patch, d_model] -> [B, N, num_patch, d_model]
        enc_out = enc_out.reshape(B, N, self.num_patch, self.d_model)

        # ==================== Step 7: 输出投影 ====================
        # [B, N, num_patch, d_model] -> [B, pred_len, N]
        seasonal_out = self.head(enc_out)

        # ==================== Step 8: 合并趋势和季节项 ====================
        dec_out = seasonal_out + trend_out

        # denorm
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out


if __name__ == '__main__':
    """
    测试代码
    """


    class Configs:
        seq_len = 96
        pred_len = 96
        enc_in = 7
        c_out = 7
        d_model = 64
        n_heads = 8
        d_ff = 128
        e_layers = 2
        dropout = 0.1
        activation = 'gelu'
        moving_avg = 25
        output_attention = False
        # Patch参数
        patch_len = 16
        stride = 8
        # Fourier参数
        mode_select = 'random'
        modes = 32


    configs = Configs()
    model = Model(configs)

    print('=' * 50)
    print('模型参数量: {:,}'.format(sum(p.numel() for p in model.parameters())))
    print('=' * 50)

    # 测试前向传播
    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.seq_len // 2 + configs.pred_len, configs.enc_in)
    x_mark_dec = torch.randn(batch_size, configs.seq_len // 2 + configs.pred_len, 4)

    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print(f'输入shape: {x_enc.shape}')
    print(f'输出shape: {out.shape}')
    print(f'期望shape: [{batch_size}, {configs.pred_len}, {configs.enc_in}]')
    assert out.shape == (batch_size, configs.pred_len, configs.enc_in), "Shape不匹配!"
    print('✓ 测试通过!')