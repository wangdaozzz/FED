import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.PatchEmbedding import PatchEmbedding
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm
from layers.FourierCorrelation import FourierBlock
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.RevIN import RevIN

class Model(nn.Module):
    """
    ProbModel 消融基线版：
    RevIN + 多尺度 Hierarchical Patch + 频域滤波 + 通道独立 + 【传统 Linear 多分位数预测】
    (不包含 KAN 模块，用于对比证明 KAN 在非线性尾部概率拟合上的优势)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # 1. 稳健的基础设施: 可逆实例归一化，解决分布漂移
        self.revin = RevIN(num_features=configs.enc_in, affine=True, subtract_last=False)

        # ==================== 核心创新 A: 多尺度 Patch Embedding ====================
        # 定义三种不同视野尺度的 Patch 
        
        # 小尺度 (精细高频视野)：patch_len=16, stride=8
        self.patch_embed_small = PatchEmbedding(
            seq_len=configs.seq_len, 
            patch_len=16, 
            stride=8, 
            d_model=configs.d_model, 
            dropout=configs.dropout
        )
        
        # 中尺度 (中程视野)：patch_len=32, stride=16
        self.patch_embed_mid = PatchEmbedding(
            seq_len=configs.seq_len, 
            patch_len=32, 
            stride=16, 
            d_model=configs.d_model, 
            dropout=configs.dropout
        )
        
        # 大尺度 (宏观长程视野)：patch_len=64, stride=32
        self.patch_embed_large = PatchEmbedding(
            seq_len=configs.seq_len, 
            patch_len=64, 
            stride=32, 
            d_model=configs.d_model, 
            dropout=configs.dropout
        )

        # 多尺度自适应融合权重
        self.agg_weights = nn.Parameter(torch.ones(3))
        
        # ==================== 核心主干: FFT 频域去噪编码器 ====================
        # 严格按照滑动窗口切块公式计算小尺度 Patch 产生的数量 M_small
        self.M_small = int((configs.seq_len - 16) / 8 + 1)

        encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                        out_channels=configs.d_model,
                                        seq_len=self.M_small, 
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

        # ==================== 传统线性多分位数概率输出头 ====================
        self.flatten = nn.Flatten(start_dim=-2)
        
        # 设定输出 3 个分位点 (例如 0.1, 0.5, 0.9)
        self.num_quantiles = 3 
        
        # 弃用 KAN，改用传统的 nn.Linear。
        # 输出维度为：预测长度 * 分位点数量
        self.predict_linear = nn.Linear(
            self.M_small * configs.d_model, 
            configs.pred_len * self.num_quantiles
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc shape: [Batch, Seq_len, Vars] 
        B, L, N = x_enc.shape
        
        # 1. RevIN 归一化 (抗分布漂移)
        x_enc = self.revin(x_enc, 'norm')

        # ==================== 多尺度特征提取 ====================
        out_s = self.patch_embed_small(x_enc)  # [B, N, M_small, D]
        out_m = self.patch_embed_mid(x_enc)    # [B, N, M_mid, D]
        out_l = self.patch_embed_large(x_enc)  # [B, N, M_large, D]
        
        # ==================== 尺度对齐与聚合 ====================
        # 对齐 out_m (中尺度)
        out_m_flat = out_m.reshape(B * N, out_m.shape[2], self.d_model).transpose(-1, -2) # [B*N, D, M_mid]
        out_m_aligned = F.interpolate(out_m_flat, size=self.M_small, mode='linear', align_corners=False)
        out_m_aligned = out_m_aligned.transpose(-1, -2).reshape(B, N, self.M_small, self.d_model) # [B, N, M_small, D]
        
        # 对齐 out_l (大尺度)
        out_l_flat = out_l.reshape(B * N, out_l.shape[2], self.d_model).transpose(-1, -2) # [B*N, D, M_large]
        out_l_aligned = F.interpolate(out_l_flat, size=self.M_small, mode='linear', align_corners=False)
        out_l_aligned = out_l_aligned.transpose(-1, -2).reshape(B, N, self.M_small, self.d_model) # [B, N, M_small, D]
        
        # 计算 softmax 权重并加权融合
        w = F.softmax(self.agg_weights, dim=0)
        enc_in = w[0] * out_s + w[1] * out_m_aligned + w[2] * out_l_aligned
        
        # ==================== 频域去噪与特征解耦 ====================
        # 通道独立 (CI) 重塑
        enc_in = enc_in.reshape(B * N, self.M_small, self.d_model)

        # 频域特征提纯
        enc_out, attns = self.encoder(enc_in, attn_mask=None)
        
        # ==================== 线性输出预测映射 ====================
        # 还原维度，准备压扁
        enc_out = enc_out.reshape(B, N, self.M_small, self.d_model)
        dec_out = self.flatten(enc_out) # shape: [B, N, M_small * d_model]
        
        # 传统线性映射
        dec_out = self.predict_linear(dec_out) # shape: [B, N, pred_len * num_quantiles]
        
        # 将输出重塑为概率预测要求的形状: [Batch, Vars, pred_len, num_quantiles]
        dec_out = dec_out.reshape(B, N, self.pred_len, self.num_quantiles)
        
        # 转换为标准输出形状: [Batch, pred_len, Vars, num_quantiles]
        dec_out = dec_out.transpose(1, 2) 
        
        # ==================== 最终反归一化还原量纲 ====================
        # 针对每个分位数输出分别进行 RevIN 反归一化
        dec_out_list = []
        for q in range(self.num_quantiles):
            # 取出当前分位数的预测线
            q_out = dec_out[..., q] # shape: [Batch, pred_len, Vars]
            # 进行 RevIN 的反归一化
            q_out = self.revin(q_out, 'denorm')
            dec_out_list.append(q_out.unsqueeze(-1))
            
        # 重新拼接成 4D 张量返回
        dec_out = torch.cat(dec_out_list, dim=-1) # shape: [Batch, pred_len, Vars, num_quantiles]

        return dec_out