import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.FourierCorrelation import FourierBlock
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.AutoCorrelation import AutoCorrelationLayer


class Model(nn.Module):
    """
    Ablation Study: w/o Channel Independence (No CI)

    特征:
    1. 保留 Patching 机制。
    2. 移除 Channel Independence：所有变量混合在一个 Patch 向量中处理。
    3. Embedding 映射：[patch_len * enc_in] -> [d_model]。
    4. Batch Size 保持为 B，而不是 B * N。
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

        # ==================== Patch 参数 ====================
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)

        # 计算 Patch 数量 (这里简化计算，假设做了适当 Padding 或整除)
        # 实际逻辑通常包含 Padding，这里参考标准计算
        self.num_patch = (self.seq_len - self.patch_len) // self.stride + 1
        print(f'w/o CI Model: patch_len={self.patch_len}, stride={self.stride}, num_patch={self.num_patch}')

        # ==================== Patch Projection (No CI 核心) ====================
        # 关键修改：输入维度包含了所有变量 (patch_len * enc_in)
        # 这一步实现了 Channel Mixing
        self.patch_proj = nn.Linear(self.patch_len * self.enc_in, self.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # ==================== 分解模块 ====================
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # ==================== 频域 Attention ====================
        # 保持在 Patch 维度做频域分析
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 32)
        enc_modes = int(min(self.modes, self.num_patch // 2))

        encoder_self_att = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.num_patch,
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
        # 头部需要一次性预测所有变量
        # 输入: [B, num_patch * d_model]
        # 输出: [B, pred_len * c_out]
        self.head = nn.Linear(self.num_patch * self.d_model, self.pred_len * self.c_out)

        # Trend 部分 (保持简单线性，但也是一次性处理)
        # [B, L, N] -> [B, pred_len, N]
        # 注意：这里的 Linear 作用在最后一维，为了保持 per-channel trend 逻辑 (DLinear风格)，
        # 我们通常还是对时间维度做投影，这部分保持不变比较合理，
        # 或者也可以改为 Linear(seq_len * enc_in, pred_len * c_out) 彻底混合。
        # 为了控制变量，建议 Trend 保持原样 (Per-channel)，只在 Attention 部分做 No CI 对比。
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # x_enc: [Batch, Seq_len, Num_variables]
        B, L, N = x_enc.shape

        # 1. 分解
        seasonal_init, trend_init = self.decomp(x_enc)

        # 2. Trend 投影
        # [B, L, N] -> [B, N, L] -> [B, N, pred_len] -> [B, pred_len, N]
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 3. Patching Process (No CI)
        # 目标: 将 [B, L, N] 转换为 [B, num_patch, patch_len * N]

        # [B, L, N] -> [B, N, L]
        x = seasonal_init.permute(0, 2, 1)

        # 使用 unfold 生成 patch
        # x.unfold(dimension, size, step) -> [B, N, num_patch, patch_len]
        x_patched = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # 调整维度以混合变量
        # [B, N, num_patch, patch_len] -> [B, num_patch, N, patch_len]
        x_patched = x_patched.permute(0, 2, 1, 3)

        # Flatten N 和 patch_len，实现 Channel Mixing
        # -> [B, num_patch, N * patch_len]
        x_patched = x_patched.reshape(B, self.num_patch, N * self.patch_len)

        # 4. Projection
        # [B, num_patch, d_model]
        enc_out = self.dropout(self.patch_proj(x_patched))

        # 5. Encoder (频域 Attention)
        # 注意：这里的 Batch Size 是 B，不是 B*N
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 6. Head
        # Flatten patch 维度
        # [B, num_patch, d_model] -> [B, num_patch * d_model]
        enc_out = enc_out.reshape(B, -1)

        # Projection -> [B, pred_len * c_out]
        seasonal_out = self.head(enc_out)

        # Reshape -> [B, pred_len, c_out]
        seasonal_out = seasonal_out.reshape(B, self.pred_len, self.c_out)

        # 7. Add Trend
        dec_out = seasonal_out + trend_out

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out