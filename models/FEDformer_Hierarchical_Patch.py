import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.PatchEmbedding import PatchEmbedding, FlattenHead
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.AutoCorrelation import AutoCorrelationLayer

class Model(nn.Module):
    """
    FEP-Former 进阶版：多尺度层次化分块 + 频域滤波 + 通道独立
    融合了 2026 年最新 PatchFormer 的 Hierarchical 思想。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # ==================== 核心创新 1: 多尺度 Patch Embedding ====================
        # 定义三种不同视野尺度的 Patch (模拟 Hierarchical 机制)
        # 严格对齐您的 layers/PatchEmbedding.py 参数签名: 
        # __init__(self, seq_len, patch_len, stride, d_model, dropout)
        
        # 1. 小尺度 (精细高频视野)：patch_len=16, stride=8
        self.patch_embed_small = PatchEmbedding(
            seq_len=configs.seq_len, 
            patch_len=16, 
            stride=8, 
            d_model=configs.d_model, 
            dropout=configs.dropout
        )
        
        # 2. 中尺度 (中程视野)：patch_len=32, stride=16
        self.patch_embed_mid = PatchEmbedding(
            seq_len=configs.seq_len, 
            patch_len=32, 
            stride=16, 
            d_model=configs.d_model, 
            dropout=configs.dropout
        )
        
        # 3. 大尺度 (宏观长程视野)：patch_len=64, stride=32
        self.patch_embed_large = PatchEmbedding(
            seq_len=configs.seq_len, 
            patch_len=64, 
            stride=32, 
            d_model=configs.d_model, 
            dropout=configs.dropout
        )

        # 多尺度自适应融合权重 (可学习参数)
        # 初始化为全 1，通过 softmax 实现动态比例分配
        self.agg_weights = nn.Parameter(torch.ones(3))
        
       # ==================== 核心主干: FFT 频域编码器 ====================
        encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                        out_channels=configs.d_model,
                                        seq_len=int(self.seq_len / 8), # 小尺度 patch 数量
                                        modes=configs.modes,
                                        mode_select_method=configs.mode_select)

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

        # ==================== 极简展平头 (Flatten Head) ====================
        self.flatten = nn.Flatten(start_dim=-2)
        
        # 严格按照滑动窗口切块公式计算 M_small 的数量: (seq_len - patch_len) / stride + 1
        # 小尺度的参数为: patch_len=16, stride=8
        M_small = int((configs.seq_len - 16) / 8 + 1) 
        
        # 动态初始化正确维度的线性层
        self.predict_linear = nn.Linear(M_small * configs.d_model, configs.pred_len)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 进来的数据维度 x_enc: [Batch, Seq_len, Vars] 
        B, L, N = x_enc.shape
        
        # 1. 均值归一化 (抗分布漂移，可选，如果加了 RevIN 这里会用到)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # ==================== 多尺度特征提取与通道独立 (CI) ====================
        # PatchEmbedding 内部会自动把 [B, L, N] 转置为处理通道独立的形状
        
        # (1) 小尺度特征 -> shape: [B, N, M_small, d_model]
        out_s = self.patch_embed_small(x_enc)
        M_small = out_s.shape[-2]
        
        # (2) 中尺度特征 -> shape: [B, N, M_mid, d_model]
        out_m = self.patch_embed_mid(x_enc)
        
        # (3) 大尺度特征 -> shape: [B, N, M_large, d_model]
        out_l = self.patch_embed_large(x_enc)
        
        # ==================== 尺度对齐与聚合 (Aggregation) ====================
        # 因为不同步长切出来的 Patch 数量不一样 (M_small > M_mid > M_large)
        # 需要在时间序列长度维度上进行插值对齐。
        # 注意：F.interpolate(mode='linear') 仅支持 3D 输入 [B, C, L]，因此需要临时将 Batch 和 N 合并
        
        # 1. 对齐 out_m (中尺度)
        out_m_flat = out_m.reshape(B * N, out_m.shape[2], self.d_model).transpose(-1, -2) # [B*N, D, M_mid]
        out_m_aligned = F.interpolate(out_m_flat, size=M_small, mode='linear', align_corners=False)
        out_m_aligned = out_m_aligned.transpose(-1, -2).reshape(B, N, M_small, self.d_model) # 变回 [B, N, M_small, D]
        
        # 2. 对齐 out_l (大尺度)
        out_l_flat = out_l.reshape(B * N, out_l.shape[2], self.d_model).transpose(-1, -2) # [B*N, D, M_large]
        out_l_aligned = F.interpolate(out_l_flat, size=M_small, mode='linear', align_corners=False)
        out_l_aligned = out_l_aligned.transpose(-1, -2).reshape(B, N, M_small, self.d_model) # 变回 [B, N, M_small, D]
        
        # 3. 计算 softmax 权重，保证权重和为 1
        w = F.softmax(self.agg_weights, dim=0)
        
        # 4. 加权融合出最终的富含多尺度语义的输入特征
        enc_in = w[0] * out_s + w[1] * out_m_aligned + w[2] * out_l_aligned
        
        # ==================== 通道独立 (CI) 重塑 ====================
        # 把 Batch(B) 和 Vars(N) 合并，送入同一个频域网络去噪，避免虚假耦合
        enc_in = enc_in.reshape(B * N, M_small, self.d_model)

        # ==================== 频域去噪主干 ====================
        # encoder 的输入要求是 [B*N, M, D]
        enc_out, attns = self.encoder(enc_in, attn_mask=None)
        
        # ==================== 极简展平头映射 ====================
        # enc_out 形状为 [B*N, M_small, d_model]
        # 还原回 B 和 N 分离的状态
        enc_out = enc_out.reshape(B, N, M_small, self.d_model)
        
        # 将最后两个维度 (M_small, d_model) 压扁
        dec_out = self.flatten(enc_out) # shape: [B, N, M_small * d_model]
        
        # 线性映射到未来预测长度
        dec_out = self.predict_linear(dec_out) # shape: [B, N, pred_len]
        
        # 将序列维度调回标准输出格式 [B, pred_len, N]
        dec_out = dec_out.transpose(-1, -2)
        
        # 反归一化 (加回均值和方差)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out # shape: [B, pred_len, N]