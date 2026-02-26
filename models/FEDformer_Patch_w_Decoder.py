import torch
import torch.nn as nn
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
# 导入两种 Layer：Encoder 用 Autoformer 的(含分解)，Decoder 用 Transformer 的(纯净版)
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.Transformer_EncDec import Decoder, DecoderLayer
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.PatchEmbedding import PatchEmbedding


class Model(nn.Module):
    """
    Ablation Study: w/ Decoder

    特征:
    1. 保留 Patching + CI。
    2. 恢复 Encoder-Decoder 结构。
    3. Decoder 输入也经过 Patching，并在 Latent Space 做 Cross Attention。
    4. Trend 依然独立处理。
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ==================== 基本参数 ====================
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len  # 需要 label_len 来构建 Decoder 输入
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention

        # ==================== Patch 参数 ====================
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)

        # 1. Encoder Patch Embedding
        self.enc_patch_embedding = PatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            dropout=configs.dropout
        )
        self.num_patch_enc = self.enc_patch_embedding.num_patch

        # 2. Decoder Patch Embedding
        # Decoder 输入长度 = label_len + pred_len
        self.dec_seq_len = self.label_len + self.pred_len
        self.dec_patch_embedding = PatchEmbedding(
            seq_len=self.dec_seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            dropout=configs.dropout
        )
        self.num_patch_dec = self.dec_patch_embedding.num_patch

        print(f'w/ Decoder Model: enc_patches={self.num_patch_enc}, dec_patches={self.num_patch_dec}')

        # ==================== 分解模块 ====================
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # ==================== 频域 Attention 配置 ====================
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 32)

        # Encoder Modes
        enc_modes = int(min(self.modes, self.num_patch_enc // 2))
        # Decoder Modes (因为序列长度可能不同，单独计算)
        dec_modes = int(min(self.modes, self.num_patch_dec // 2))

        # 1. Encoder Self Attention
        encoder_self_att = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.num_patch_enc,
            modes=enc_modes,
            mode_select_method=self.mode_select
        )

        # 2. Decoder Self Attention
        decoder_self_att = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.num_patch_dec,
            modes=dec_modes,
            mode_select_method=self.mode_select
        )

        # 3. Decoder Cross Attention
        decoder_cross_att = FourierCrossAttention(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len_q=self.num_patch_dec,
            seq_len_kv=self.num_patch_enc,
            modes=dec_modes,  # 通常以 Query 长度为准
            mode_select_method=self.mode_select
        )

        # ==================== Encoder & Decoder ====================
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

        self.decoder = Decoder(
            [
                DecoderLayer(
                    # 这里的 Self/Cross Attention 都需要用 AutoCorrelationLayer 包装
                    AutoCorrelationLayer(decoder_self_att, self.d_model, configs.n_heads),
                    AutoCorrelationLayer(decoder_cross_att, self.d_model, configs.n_heads),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
            projection=None  # 我们自己手动投影
        )

        # ==================== 输出投影层 ====================
        # 将 Decoder 输出的 Patch 特征还原回 Patch 数据点
        # [d_model] -> [patch_len]
        self.dec_head = nn.Linear(self.d_model, self.patch_len)

        # Trend 投影
        self.trend_projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # x_enc: [Batch, Seq_len, Num_variables]
        B, L, N = x_enc.shape

        # 1. 分解与 Trend
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. Encoder 处理 (Patch + CI)
        enc_out = self.enc_patch_embedding(seasonal_init)  # [B, N, num_patch, D]
        enc_out = enc_out.reshape(B * N, self.num_patch_enc, self.d_model)  # CI
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 3. 构造 Decoder 输入
        # 传统的 Transformer 做法: Label (后半段) + Zeros (预测段)
        seasonal_label = seasonal_init[:, -self.label_len:, :]
        zeros = torch.zeros((B, self.pred_len, N), device=x_enc.device)
        dec_input = torch.cat([seasonal_label, zeros], dim=1)  # [B, label_len+pred_len, N]

        # 4. Decoder Patching + CI
        dec_out = self.dec_patch_embedding(dec_input)  # [B, N, num_patch_dec, D]
        dec_out = dec_out.reshape(B * N, self.num_patch_dec, self.d_model)

        # 5. Decoder Forward
        # 输入: dec_out (Query), enc_out (Key/Value)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # 6. 还原序列
        # [B*N, num_patch_dec, D] -> [B*N, num_patch_dec, patch_len]
        dec_out = self.dec_head(dec_out)

        # Flatten 回时间序列
        # [B*N, num_patch_dec * patch_len]
        dec_out = dec_out.reshape(B * N, -1)

        # 截取最后的 pred_len 长度 (Patching 可能会有 Padding，所以长度可能略大于 dec_seq_len)
        # 我们只需要最后 pred_len 个点
        dec_out = dec_out[:, -self.pred_len:]

        # Reshape 回 [B, pred_len, N]
        dec_out = dec_out.reshape(B, N, self.pred_len).permute(0, 2, 1)

        # 7. Add Trend
        final_out = dec_out + trend_out

        if self.output_attention:
            return final_out, attns
        else:
            return final_out