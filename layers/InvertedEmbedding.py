import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InvertedEmbedding(nn.Module):
    """
    Inverted Embedding: 将每个变量的整个时间序列映射到d_model维度

    传统Embedding: [B, L, N] -> 每个时间点映射 -> [B, L, d_model]
    Inverted Embedding: [B, L, N] -> 每个变量映射 -> [B, N, d_model]

    Input:  [Batch, Seq_len, Num_variables]
    Output: [Batch, Num_variables, d_model]
    """

    def __init__(self, seq_len, d_model, dropout=0.1):
        super(InvertedEmbedding, self).__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # 将整个时间序列映射到d_model维度
        self.value_embedding = nn.Linear(seq_len, d_model)

        # 可学习的位置编码（变量位置）
        self.position_embedding = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        self.dropout = nn.Dropout(dropout)

        print(f'[InvertedEmbedding] seq_len={seq_len} -> d_model={d_model}')

    def forward(self, x):
        """
        x: [B, L, N]
        return: [B, N, d_model]
        """
        # [B, L, N] -> [B, N, L]
        x = x.permute(0, 2, 1)

        # [B, N, L] -> [B, N, d_model]
        x = self.value_embedding(x)

        # 添加位置编码
        x = x + self.position_embedding

        return self.dropout(x)


class InvertedOutputHead(nn.Module):
    """
    Inverted输出头: 将d_model映射回pred_len

    Input:  [Batch, Num_variables, d_model]
    Output: [Batch, Pred_len, Num_variables]
    """

    def __init__(self, d_model, pred_len, dropout=0.1):
        super(InvertedOutputHead, self).__init__()

        self.d_model = d_model
        self.pred_len = pred_len

        self.linear = nn.Linear(d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

        print(f'[InvertedOutputHead] d_model={d_model} -> pred_len={pred_len}')

    def forward(self, x):
        """
        x: [B, N, d_model]
        return: [B, pred_len, N]
        """
        # [B, N, d_model] -> [B, N, pred_len]
        x = self.linear(x)
        x = self.dropout(x)

        # [B, N, pred_len] -> [B, pred_len, N]
        x = x.permute(0, 2, 1)

        return x


if __name__ == '__main__':
    """测试代码"""
    batch_size = 4
    seq_len = 96
    pred_len = 96
    num_vars = 8
    d_model = 512

    # 测试 InvertedEmbedding
    embed = InvertedEmbedding(seq_len=seq_len, d_model=d_model)
    x = torch.randn(batch_size, seq_len, num_vars)
    out = embed(x)
    print(f'输入: {x.shape}')
    print(f'Embedding输出: {out.shape}')

    # 测试 InvertedOutputHead
    head = InvertedOutputHead(d_model=d_model, pred_len=pred_len)
    final = head(out)
    print(f'最终输出: {final.shape}')

    print('✓ 测试通过!')