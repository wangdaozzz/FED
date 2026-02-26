import torch
import torch.nn as nn
from layers.SelfAttention_Family import FullAttention  # 关键：导入标准的时域注意力
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.AutoCorrelation import AutoCorrelationLayer
# 假设你已经有了这两个自定义层（和你之前的代码一致）
from layers.PatchEmbedding import PatchEmbedding, FlattenHead


class Model(nn.Module):
    """
    Ablation Study: w/o Frequency Attention (Time Domain)

    特征:
    1. 保留 Patching 机制。
    2. 保留 Channel Independence (CI) 策略。
    3. 核心修改：将 FourierBlock 替换为 FullAttention。
       这意味着 Attention 计算发生在 Patch 的时域上，而不是频域上。
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ==================== 基本参数 ====================
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model

        # ==================== Patch 参数 ====================
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)

        # Patch Embedding (保留，含 CI 处理准备)
        self.patch_embedding = PatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            dropout=configs.dropout
        )
        self.num_patch = self.patch_embedding.num_patch
        print(f'w/o Freq Attn Model: patch_len={self.patch_len}, num_patch={self.num_patch}')

        # ==================== 分解模块 (保留) ====================
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # ==================== Attention 替换核心 ====================
        # 关键修改：不再使用 FourierBlock，而是初始化 FullAttention
        # FullAttention 是标准 Transformer 的计算方式：Softmax(QK^T/d)V

        encoder_self_att = FullAttention(
            mask_flag=False,  # Encoder 通常不需要 mask
            factor=configs.factor,
            scale=None,
            attention_dropout=configs.dropout,
            output_attention=configs.output_attention
        )

        print('Ablation Info: Replacing FourierBlock with Vanilla FullAttention (Time Domain)')

        # ==================== Encoder ====================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # 传入时域 Attention 实例
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

        # ==================== 输出投影层 (保留 Patch Flatten) ====================
        self.head = FlattenHead(
            num_patch=self.num_patch,
            d_model=self.d_model,
            pred_len=self.pred_len,
            dropout=configs.dropout
        )

        # Trend 部分 (保留 DLinear 风格)
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # x_enc: [Batch, Seq_len, Num_variables]
        B, L, N = x_enc.shape

        # 1. 序列分解
        seasonal_init, trend_init = self.decomp(x_enc)

        # 2. Trend 处理
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 3. Seasonal 处理 - Patching
        # [B, L, N] -> [B, N, num_patch, d_model]
        enc_out = self.patch_embedding(seasonal_init)

        # 4. Channel Independence (Reshape)
        # [B, N, num_patch, d_model] -> [B*N, num_patch, d_model]
        enc_out = enc_out.reshape(B * N, self.num_patch, self.d_model)

        # 5. Encoder (现在执行的是时域 Full Attention)
        # 输入维度: [B*N, num_patch, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 6. 恢复维度
        # [B*N, num_patch, d_model] -> [B, N, num_patch, d_model]
        enc_out = enc_out.reshape(B, N, self.num_patch, self.d_model)

        # 7. Flatten Head 预测
        seasonal_out = self.head(enc_out)

        # 8. 最终合并
        dec_out = seasonal_out + trend_out

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out