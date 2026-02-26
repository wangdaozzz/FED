import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 导入原有的FEDformer组件
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.AutoCorrelation import AutoCorrelationLayer

# 导入Patch组件
from layers.PatchEmbedding import PatchEmbedding, FlattenHead


class Model(nn.Module):
    """
    FEDformer + Patching + Inverted

    核心创新：
    1. Patching: 将时间序列分割为patch，降低序列长度
    2. Inverted: 在变量维度做attention，捕获变量间相关性
    3. 双阶段Attention:
       - Stage 1: Temporal Attention (在patch维度)
       - Stage 2: Variate Attention (在变量维度)
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ==================== 基本参数 ====================
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 输入变量数
        self.c_out = configs.c_out  # 输出变量数
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.output_attention = configs.output_attention

        # ==================== Patch参数 ====================
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)

        # ==================== Patch Embedding ====================
        self.patch_embedding = PatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            dropout=self.dropout
        )

        # 从patch_embedding获取实际的num_patch
        self.num_patch = self.patch_embedding.num_patch

        print(f'[FEDformer_Patch_Inv] 配置信息:')
        print(f'  - seq_len={self.seq_len}, pred_len={self.pred_len}')
        print(f'  - patch_len={self.patch_len}, stride={self.stride}, num_patch={self.num_patch}')
        print(f'  - enc_in (变量数)={self.enc_in}')
        print(f'  - d_model={self.d_model}, n_heads={self.n_heads}, e_layers={self.e_layers}')

        # ==================== 分解模块 ====================
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, str):
            kernel_size = int(kernel_size)
        if isinstance(kernel_size, list) and len(kernel_size) == 1:
            kernel_size = kernel_size[0]

        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # ==================== Fourier参数 ====================
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 32)

        # ==================== Stage 1: Temporal Encoder ====================
        # 在patch维度做频域attention
        temporal_modes = int(min(self.modes, self.num_patch // 2))
        print(f'  - Temporal FourierBlock modes={temporal_modes}')

        temporal_self_att = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.num_patch,  # patch数量作为序列长度
            modes=temporal_modes,
            mode_select_method=self.mode_select
        )

        self.temporal_encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        temporal_self_att,
                        self.d_model,
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    moving_avg=kernel_size if isinstance(kernel_size, int) else kernel_size[0],
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )

        # ==================== Stage 2: Variate Encoder (新增!) ====================
        # 在变量维度做频域attention
        variate_modes = int(min(self.modes, self.enc_in // 2))
        # 确保至少有1个mode
        variate_modes = max(variate_modes, 1)
        print(f'  - Variate FourierBlock modes={variate_modes}')

        variate_self_att = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.enc_in,  # 变量数量作为序列长度
            modes=variate_modes,
            mode_select_method=self.mode_select
        )

        self.variate_encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        variate_self_att,
                        self.d_model,
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    moving_avg=kernel_size if isinstance(kernel_size, int) else kernel_size[0],
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )

        # ==================== 融合层 (可选) ====================
        # 用于融合temporal和variate两个分支的信息
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 可学习的融合权重

        # ==================== 输出投影层 ====================
        # Seasonal部分: Patch展平后投影
        self.head = FlattenHead(
            num_patch=self.num_patch,
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

        # ==================== Step 2: Trend部分处理 ====================
        # [B, L, N] -> [B, N, L] -> [B, N, pred_len] -> [B, pred_len, N]
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1))
        trend_out = trend_out.permute(0, 2, 1)

        # ==================== Step 3: Patch Embedding ====================
        # [B, L, N] -> [B, N, num_patch, d_model]
        enc_out = self.patch_embedding(seasonal_init)

        # ==================== Step 4: Stage 1 - Temporal Attention ====================
        # 在patch维度做attention，每个变量独立处理 (Channel Independence)
        # [B, N, num_patch, d_model] -> [B*N, num_patch, d_model]
        temporal_in = enc_out.reshape(B * N, self.num_patch, self.d_model)
        temporal_out, _ = self.temporal_encoder(temporal_in, attn_mask=enc_self_mask)
        # [B*N, num_patch, d_model] -> [B, N, num_patch, d_model]
        temporal_out = temporal_out.reshape(B, N, self.num_patch, self.d_model)

        # ==================== Step 5: Stage 2 - Variate Attention (新增!) ====================
        # 在变量维度做attention，每个patch独立处理
        # [B, N, num_patch, d_model] -> [B, num_patch, N, d_model]
        variate_in = enc_out.permute(0, 2, 1, 3)
        # [B, num_patch, N, d_model] -> [B*num_patch, N, d_model]
        variate_in = variate_in.reshape(B * self.num_patch, N, self.d_model)
        variate_out, _ = self.variate_encoder(variate_in, attn_mask=enc_self_mask)
        # [B*num_patch, N, d_model] -> [B, num_patch, N, d_model]
        variate_out = variate_out.reshape(B, self.num_patch, N, self.d_model)
        # [B, num_patch, N, d_model] -> [B, N, num_patch, d_model]
        variate_out = variate_out.permute(0, 2, 1, 3)

        # ==================== Step 6: 融合两个分支 ====================
        # 方式1: 可学习权重融合
        fusion_weight = torch.sigmoid(self.fusion_weight)  # 归一化到0-1
        enc_out = fusion_weight * temporal_out + (1 - fusion_weight) * variate_out

        # 方式2: 残差连接 (备选)
        # enc_out = temporal_out + variate_out

        # 方式3: 拼接后投影 (备选，需要额外的投影层)
        # enc_out = torch.cat([temporal_out, variate_out], dim=-1)
        # enc_out = self.fusion_proj(enc_out)

        # ==================== Step 7: 输出投影 ====================
        # [B, N, num_patch, d_model] -> [B, pred_len, N]
        seasonal_out = self.head(enc_out)

        # ==================== Step 8: 合并趋势和季节项 ====================
        dec_out = seasonal_out + trend_out

        if self.output_attention:
            return dec_out, None
        else:
            return dec_out


if __name__ == '__main__':
    """
    测试代码
    """


    class Configs:
        seq_len = 96
        pred_len = 96
        enc_in = 8  # Exchange数据集8个变量
        c_out = 8
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

    print('=' * 60)
    print('模型参数量: {:,}'.format(sum(p.numel() for p in model.parameters())))
    print('=' * 60)

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

    # 打印融合权重
    print(f'融合权重 (sigmoid): {torch.sigmoid(model.fusion_weight).item():.4f}')