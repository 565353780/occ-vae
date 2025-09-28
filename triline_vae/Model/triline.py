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

    def queryPoints(self, coords: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
        """
        feats: [B, 3, feat_num, feat_dim]
        positions: [B, 3, feat_num], sorted ascending in [-0.5, 0.5]
        coords: [B, M, 3], query points in [-0.5, 0.5]

        returns:
        features: [B, M, 3, feat_dim]  # 三个轴的插值特征分别保留
        """
        B, _, feat_num, feat_dim = self.feats.shape
        M = coords.shape[1]

        # reshape for axis-wise processing
        feats_reshape = self.feats.view(
            B * 3, feat_num, feat_dim
        )  # [B*3, feat_num, feat_dim]
        positions_reshape = self.positions.view(B * 3, feat_num)  # [B*3, feat_num]
        coords_reshape = coords.permute(0, 2, 1).reshape(B * 3, M)  # [B*3, M]

        outputs = []
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            coords_chunk = coords_reshape[:, start:end]  # [B*3, m]

            # searchsorted
            idx = torch.searchsorted(
                positions_reshape, coords_chunk, right=True
            )  # [B*3, m]

            # 边界处理
            mask_left = coords_chunk <= positions_reshape[:, :1]
            idx[mask_left] = 1
            mask_right = coords_chunk >= positions_reshape[:, -1:]
            idx[mask_right] = feat_num - 1
            idx = idx.clamp(1, feat_num - 1)

            idx0 = idx - 1
            idx1 = idx

            # gather feats
            feat0 = feats_reshape.gather(1, idx0.unsqueeze(-1).expand(-1, -1, feat_dim))
            feat1 = feats_reshape.gather(1, idx1.unsqueeze(-1).expand(-1, -1, feat_dim))

            # gather positions
            pos0 = positions_reshape.gather(1, idx0)
            pos1 = positions_reshape.gather(1, idx1)

            # linear interp
            denom = (pos1 - pos0).clamp(min=1e-8)
            weight = (coords_chunk - pos0) / denom
            weight = weight.unsqueeze(-1)  # [B*3, m, 1]

            feat_interp = feat0 * (1 - weight) + feat1 * weight  # [B*3, m, feat_dim]
            outputs.append(feat_interp)

        # 拼接结果
        feat_interp_all = torch.cat(outputs, dim=1)  # [B*3, M, feat_dim]
        feat_interp_all = feat_interp_all.view(B, 3, M, feat_dim).permute(0, 2, 1, 3)
        return feat_interp_all

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
