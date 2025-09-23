import torch
from typing import Union

from triline_vae.Method.position import deltas_to_positions


class Triline(object):
    def __init__(
        self,
        feats: torch.Tensor,  # [B, 3, feat_num, feat_dim]
        deltas: Union[torch.Tensor, None] = None,  # [B, 3, feat_num - 1]
    ) -> None:
        self.feats = feats

        if deltas is None:
            deltas = torch.zeros(feats.shape[0], feats.shape[1], feats.shape[2] - 1).to(
                feats.device, dtype=feats.dtype
            )
        self.positions = deltas_to_positions(deltas)
        return

    def queryPoints(self, coords: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, 3, feat_num, feat_dim]
        positions: [B, 3, feat_num], positions along each axis, sorted ascending in [-0.5, 0.5]
        coords: [B, M, 3], query points in [-0.5, 0.5]

        returns:
        features: [B, M, 3, feat_dim]  # 三个轴的插值特征分别保留
        """
        B, _, feat_num, feat_dim = self.feats.shape
        M = coords.shape[1]

        # 把B和3轴维度合并，方便searchsorted和gather同时做
        feats_reshape = self.feats.view(
            B * 3, feat_num, feat_dim
        )  # [B*3, feat_num, feat_dim]
        positions_reshape = self.positions.view(B * 3, feat_num)  # [B*3, feat_num]
        coords_reshape = coords.permute(0, 2, 1).reshape(
            B * 3, M
        )  # [B*3, M]，把轴维放在batch维合并

        # searchsorted
        idx = torch.searchsorted(
            positions_reshape, coords_reshape, right=True
        )  # [B*3, M]
        idx = idx.clamp(1, feat_num - 1)

        idx0 = idx - 1
        idx1 = idx

        # gather indices 扩展
        idx0_exp = idx0.unsqueeze(-1).expand(-1, -1, feat_dim)  # [B*3, M, feat_dim]
        idx1_exp = idx1.unsqueeze(-1).expand(-1, -1, feat_dim)

        feat0 = torch.gather(feats_reshape, 1, idx0_exp)  # [B*3, M, feat_dim]
        feat1 = torch.gather(feats_reshape, 1, idx1_exp)  # [B*3, M, feat_dim]

        idx0_pos = torch.gather(positions_reshape, 1, idx0)  # [B*3, M]
        idx1_pos = torch.gather(positions_reshape, 1, idx1)  # [B*3, M]

        weight = (coords_reshape - idx0_pos) / (idx1_pos - idx0_pos + 1e-8)  # [B*3, M]
        weight = weight.unsqueeze(-1)  # [B*3, M, 1]

        feat_interp = feat0 * (1 - weight) + feat1 * weight  # [B*3, M, feat_dim]

        # 还原回 [B, 3, M, feat_dim] 再permute成 [B, M, 3, feat_dim]
        feat_interp = feat_interp.view(B, 3, M, feat_dim).permute(0, 2, 1, 3)

        return feat_interp

    def query(self, coords: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, 3, feat_num, feat_dim]
        positions: [B, 3, feat_num]
        coords: [B, M1, M2, ..., Mk, 3]

        return: [B, M1, M2, ..., Mk, 3, feat_dim]
        """
        if coords.ndim == 3:
            return self.queryPoints(coords)

        B, _, _, feat_dim = self.feats.shape
        coord_shape = coords.shape
        spatial_dims = coord_shape[1:-1]  # (M1, M2, ..., Mk)
        M = int(torch.tensor(spatial_dims).prod().item())  # flatten所有空间维度的大小

        # reshape coords 为 [B, M, 3]
        coords_flat = coords.view(B, M, 3)

        feat_interp = self.queryPoints(coords_flat)

        # 最后把M维度reshape回原来的多维空间维度 [M1, M2, ..., Mk]
        out_shape = (B, *spatial_dims, 3, feat_dim)
        feat_interp = feat_interp.view(out_shape)

        return feat_interp
