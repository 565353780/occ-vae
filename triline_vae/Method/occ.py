import torch
from torchsparse import SparseTensor


def occ_to_torchsparse(occ: torch.Tensor) -> SparseTensor:
    coords = torch.nonzero(occ, as_tuple=False).int()

    feats = torch.ones((coords.shape[0], 1), dtype=torch.float32, device=occ.device)

    sparse_tensor = SparseTensor(coords=coords, feats=feats)
    return sparse_tensor


def make_occ_centers(
    dim: int, device: str = "cpu", dtype=torch.float32
) -> torch.Tensor:
    """
    返回 [dim^3, 3] 的中心坐标张量，定义在 [-0.5, 0.5]^3 空间，
    可reshape为 [dim, dim, dim, 3] 以匹配体素位置。
    """
    # 计算等间隔中心点位置：[-0.5 + 0.5/dim, ..., 0.5 - 0.5/dim]
    centers_1d = (
        torch.arange(dim, device=device, dtype=dtype) + 0.5
    ) / dim - 0.5  # [dim]

    # meshgrid with 'ij' indexing: [z, y, x] 形式
    z, y, x = torch.meshgrid(
        centers_1d, centers_1d, centers_1d, indexing="ij"
    )  # [dim, dim, dim]

    # 拼接并 reshape 为 [dim^3, 3]
    centers = torch.stack([x, y, z], dim=-1)  # [dim, dim, dim, 3]

    return centers
