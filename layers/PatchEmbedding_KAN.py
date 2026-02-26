import torch
import torch.nn as nn
import math
from ikan import KAN

class PatchEmbedding(nn.Module):
    """
    将时间序列分割为patch并进行嵌入
    输入: [Batch, Seq_len, Num_variables]
    输出: [Batch, Num_variables, Num_patches, d_model]
    """

    def __init__(self, seq_len, patch_len, stride, d_model, dropout):
        super(PatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # 计算需要padding的长度，确保能整除
        # 先计算不padding时能得到多少个patch
        self.num_patch_no_pad = (seq_len - patch_len) // stride + 1

        # 计算覆盖的长度
        covered_len = (self.num_patch_no_pad - 1) * stride + patch_len

        # 如果没有完全覆盖，需要padding
        if covered_len < seq_len:
            # 需要额外padding以覆盖剩余部分
            self.pad_len = stride  # padding一个stride的长度
            self.num_patch = self.num_patch_no_pad + 1
        else:
            self.pad_len = 0
            self.num_patch = self.num_patch_no_pad

        print(f'PatchEmbedding: seq_len={seq_len}, patch_len={patch_len}, stride={stride}')
        print(f'PatchEmbedding: num_patch={self.num_patch}, pad_len={self.pad_len}')

        # Patch线性投影层: 将每个patch从patch_len维度映射到d_model维度
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # 可学习的位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, 1, self.num_patch, d_model)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [Batch, Seq_len, Num_variables]
        return: [Batch, Num_variables, Num_patches, d_model]
        """
        B, L, N = x.shape

        # 转置: [B, L, N] -> [B, N, L]
        x = x.permute(0, 2, 1)

        # Padding (如果需要)
        if self.pad_len > 0:
            # 在序列末尾padding，使用复制模式
            x = nn.functional.pad(x, (0, self.pad_len), mode='replicate')

        # Patching: 使用unfold进行滑动窗口分割
        # [B, N, L_padded] -> [B, N, num_patch, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 验证patch数量
        assert x.shape[2] == self.num_patch, \
            f"Patch数量不匹配: 期望{self.num_patch}, 实际{x.shape[2]}"

        # 线性投影: [B, N, num_patch, patch_len] -> [B, N, num_patch, d_model]
        x = self.value_embedding(x)

        # 添加位置编码: position_embedding会自动广播
        x = x + self.position_embedding

        return self.dropout(x)



class FlattenHead(nn.Module):
    """
    将Encoder输出展平并投影到预测长度
    输入: [Batch, Num_variables, Num_patches, d_model]
    输出: [Batch, Pred_len, Num_variables]
    """

    def __init__(self, num_patch, d_model, pred_len, dropout):
        super(FlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)  # 展平最后两个维度
        # self.linear = nn.Linear(num_patch * d_model, pred_len)
        layers_hidden = [num_patch * d_model,1024,pred_len]
        self.linear = KAN(
            layers_hidden=layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=0.5,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [Batch, Num_variables, Num_patches, d_model]
        return: [Batch, Pred_len, Num_variables]
        """
        # [B, N, num_patch, d_model] -> [B, N, num_patch * d_model]
        x = self.flatten(x)

        # [B, N, num_patch * d_model] -> [B, N, pred_len]
        x = self.linear(x)
        x = self.dropout(x)

        # [B, N, pred_len] -> [B, pred_len, N]
        x = x.permute(0, 2, 1)

        return x


if __name__ == '__main__':
    """测试PatchEmbedding"""
    print("=" * 50)
    print("测试 PatchEmbedding")
    print("=" * 50)

    # 测试参数
    batch_size = 4
    seq_len = 96
    num_vars = 7
    patch_len = 16
    stride = 8
    d_model = 64

    # 创建模块
    patch_embed = PatchEmbedding(
        seq_len=seq_len,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        dropout=0.1
    )

    # 测试输入
    x = torch.randn(batch_size, seq_len, num_vars)
    print(f"\n输入shape: {x.shape}")

    # 前向传播
    out = patch_embed(x)
    print(f"输出shape: {out.shape}")
    print(f"期望shape: [{batch_size}, {num_vars}, {patch_embed.num_patch}, {d_model}]")

    assert out.shape == (batch_size, num_vars, patch_embed.num_patch, d_model)
    print("\n✓ PatchEmbedding测试通过!")

    print("\n" + "=" * 50)
    print("测试 FlattenHead")
    print("=" * 50)

    pred_len = 96
    head = FlattenHead(
        num_patch=patch_embed.num_patch,
        d_model=d_model,
        pred_len=pred_len,
        dropout=0.1
    )

    final_out = head(out)
    print(f"FlattenHead输出shape: {final_out.shape}")
    print(f"期望shape: [{batch_size}, {pred_len}, {num_vars}]")

    assert final_out.shape == (batch_size, pred_len, num_vars)
    print("\n✓ FlattenHead测试通过!")