import torch
from typing import Union

from occ_vae.Method.position import deltas_to_positions


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

    def query(self, coords: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, 3, feat_num, feat_dim]
        positions: [B, 3, feat_num]
        coords: [B, M1, M2, ..., Mk, 3]

        return: [B, M1, M2, ..., Mk, 3, feat_dim]
        """
        B, _, feat_num, feat_dim = self.feats.shape
        coord_shape = coords.shape
        spatial_dims = coord_shape[1:-1]  # (M1, M2, ..., Mk)
        M = int(torch.tensor(spatial_dims).prod().item())  # flatten所有空间维度的大小

        # reshape coords 为 [B, M, 3]
        coords_flat = coords.view(B, M, 3)

        # 把B和3轴合并做批量处理
        feats_reshape = self.feats.view(
            B * 3, feat_num, feat_dim
        )  # [B*3, feat_num, feat_dim]
        positions_reshape = self.positions.view(B * 3, feat_num)  # [B*3, feat_num]
        coords_reshape = coords_flat.permute(0, 2, 1).reshape(B * 3, M)  # [B*3, M]

        # 搜索插值
        idx = torch.searchsorted(positions_reshape, coords_reshape, right=True)
        idx = idx.clamp(1, feat_num - 1)

        idx0 = idx - 1
        idx1 = idx

        idx0_exp = idx0.unsqueeze(-1).expand(-1, -1, feat_dim)  # [B*3, M, feat_dim]
        idx1_exp = idx1.unsqueeze(-1).expand(-1, -1, feat_dim)

        feat0 = torch.gather(feats_reshape, 1, idx0_exp)  # [B*3, M, feat_dim]
        feat1 = torch.gather(feats_reshape, 1, idx1_exp)  # [B*3, M, feat_dim]

        idx0_pos = torch.gather(positions_reshape, 1, idx0)  # [B*3, M]
        idx1_pos = torch.gather(positions_reshape, 1, idx1)  # [B*3, M]

        weight = (coords_reshape - idx0_pos) / (idx1_pos - idx0_pos + 1e-8)  # [B*3, M]
        weight = weight.unsqueeze(-1)  # [B*3, M, 1]

        feat_interp = feat0 * (1 - weight) + feat1 * weight  # [B*3, M, feat_dim]

        # reshape回[B, 3, M, feat_dim]，再permute为 [B, M, 3, feat_dim]
        feat_interp = feat_interp.view(B, 3, M, feat_dim).permute(
            0, 2, 1, 3
        )  # [B, M, 3, feat_dim]

        # 最后把M维度reshape回原来的多维空间维度 [M1, M2, ..., Mk]
        out_shape = (B, *spatial_dims, 3, feat_dim)
        feat_interp = feat_interp.view(out_shape)

        return feat_interp
