import torch
import torch.nn as nn


class Triline(nn.Module):
    def __init__(self, N: int, C: int):
        """
        Args:
            N: 每条轴上的特征点数量
            C: 每个点的特征维度
        """
        super().__init__()
        self.N = N
        self.C = C

        # 初始化：X, Y, Z 三条轴，每条 N 个特征点，每个点是 C 维特征
        self.x_line = nn.Parameter(torch.randn(N, C))
        self.y_line = nn.Parameter(torch.randn(N, C))
        self.z_line = nn.Parameter(torch.randn(N, C))

        # 构建位置坐标 [-0.5, 0.5] 上的 N 个位置
        self.register_buffer("grid", torch.linspace(-0.5, 0.5, N))
        return

    def interpolate_1d(self, line_feat: torch.Tensor, coord: torch.Tensor):
        """
        在给定的一维特征线上，对 coord 位置进行线性插值。

        Args:
            line_feat: [N, C]
            coord: [B] 取值范围 [-0.5, 0.5]
        Returns:
            interpolated_feat: [B, C]
        """
        N = line_feat.shape[0]

        # 将 coord 映射到 grid 的索引区间
        pos = (coord - self.grid[0]) / (self.grid[1] - self.grid[0])  # → [0, N-1]
        idx0 = torch.clamp(pos.floor().long(), 0, N - 2)  # 左边界索引
        idx1 = idx0 + 1  # 右边界索引
        w1 = (pos - idx0.float()).unsqueeze(1)  # 插值权重
        w0 = 1.0 - w1

        # 获取两个点的特征值
        f0 = line_feat[idx0]  # [B, C]
        f1 = line_feat[idx1]  # [B, C]

        # 插值
        return f0 * w0 + f1 * w1

    def forward(self, coords: torch.Tensor, mode="sum"):
        """
        Args:
            coords: [B, 3]，每一行是 (x, y, z)，范围必须在 [-0.5, 0.5]
            mode: 如何组合三个方向的特征，支持 'sum'、'concat'
        Returns:
            feat: [B, C] 或 [B, 3*C]
        """
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        fx = self.interpolate_1d(self.x_line, x)
        fy = self.interpolate_1d(self.y_line, y)
        fz = self.interpolate_1d(self.z_line, z)

        if mode == "sum":
            return fx + fy + fz  # [B, C]
        elif mode == "concat":
            return torch.cat([fx, fy, fz], dim=-1)  # [B, 3*C]
        else:
            raise ValueError(f"Unknown mode: {mode}")
