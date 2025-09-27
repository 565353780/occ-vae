import torch
from torch import nn
from math import ceil
from typing import Tuple

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)

from triline_vae.Model.triline import Triline
from triline_vae.Model.point_embed import PointEmbed
from triline_vae.Model.query_fusion import QueryFusion
from triline_vae.Model.occ_decoder import OccDecoder
from triline_vae.Model.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)


class TrilineVAE(nn.Module):
    def __init__(
        self,
        feat_num: int = 2048,
        feat_dim: int = 512,
    ):
        super().__init__()
        self.feat_num = feat_num
        self.feat_dim = feat_dim

        self.latent_dim = feat_dim

        self.point_embed = PointEmbed(dim=self.latent_dim)

        self.ptv3_encoder = PointTransformerV3(self.latent_dim + 3, enc_mode=True)

        self.triline_encoder = QueryFusion(512, self.feat_num, self.latent_dim * 3)

        self.mu_fc = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_fc = nn.Linear(self.latent_dim, self.latent_dim)

        # Decoder MLP
        self.decoder = OccDecoder(
            feat_dim=feat_dim,
            hidden_dim=64,
            num_layers=5,
            use_xyz=False,
            use_posenc=False,
            posenc_freq=10,
        )
        return

    def encode(
        self, pts: torch.Tensor, deterministic: bool = False
    ) -> Tuple[Triline, torch.Tensor]:
        flatten_pts = pts.reshape(-1, pts.shape[-1])  # [B*N, C]

        # 生成 batch 索引
        batch_idxs = torch.arange(pts.shape[0], device=pts.device).repeat_interleave(
            pts.shape[1]
        )

        coord = flatten_pts[:, :3]
        feat = flatten_pts[:, 3:]

        coord_embed = self.point_embed(coord.unsqueeze(0)).squeeze(0)

        merge_feat = torch.cat([feat, coord_embed], dim=-1)

        ptv3_data = {
            "coord": coord,
            "feat": merge_feat,
            "batch": batch_idxs,
            "grid_size": 0.01,
        }

        point = self.ptv3_encoder(ptv3_data)

        batch = point.batch
        feature = point.feat

        x = self.triline_encoder(feature, batch)

        x = x.view(x.shape[0], 3, self.feat_num, self.latent_dim)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution([mu, logvar], deterministic)

        x = posterior.sample()
        kl = posterior.kl()

        triline = Triline(x)

        return triline, kl

    def decode(self, triline: Triline, coords: torch.Tensor) -> torch.Tensor:
        feat = triline.query(coords)
        feat = feat.reshape(feat.shape[:-2] + (feat.shape[-2] * feat.shape[-1],))
        occ_logits = self.decoder(feat).squeeze(-1)
        tsdf = nn.Sigmoid()(occ_logits) * 2.0 - 1.0
        return tsdf

    def decodeLarge(
        self, triline: Triline, coords: torch.Tensor, query_step_size: int = 100000
    ) -> torch.Tensor:
        if coords.shape[1] <= query_step_size:
            return self.decode(triline, coords)

        logits = []
        for block_idx in range(ceil(coords.shape[1] / query_step_size)):
            curr_logits = self.decode(
                triline,
                coords[
                    :,
                    block_idx * query_step_size : (block_idx + 1) * query_step_size,
                    :,
                ],
            )
            logits.append(curr_logits)
        logits = torch.cat(logits, dim=1)

        return logits

    def forward(self, data_dict: dict) -> dict:
        coarse_surface = data_dict["coarse_surface"]
        sharp_surface = data_dict["sharp_surface"]
        queries = data_dict["rand_points"]

        merge_surface = torch.cat([coarse_surface, sharp_surface], dim=1)

        triline, kl = self.encode(merge_surface)

        tsdf = self.decodeLarge(triline, queries)

        result_dict = {
            "tsdf": tsdf,
            "kl": kl,
        }

        return result_dict
