import torch
import torch.nn.functional as F


def deltas_to_positions(deltas: torch.Tensor) -> torch.Tensor:
    deltas = F.softplus(deltas)  # 确保正值，避免重叠
    cum = torch.cumsum(deltas, dim=-1)  # 沿最后一维累加
    total = cum[..., -1:].detach()  # 取最后一维的最后一个值，支持任意维度

    pos = cum / total  # 归一化到0~1
    pos = F.pad(pos, (1, 0), value=0.0)  # 在最后一维左侧填充0作为起点
    pos = pos * 2.0 - 1.0  # 映射到[-0.5, 0.5]

    return pos
