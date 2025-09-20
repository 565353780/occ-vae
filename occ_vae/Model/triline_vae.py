import torch
from torch import nn
from math import ceil
from typing import Tuple

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)

from occ_vae.Model.triline import Triline
from occ_vae.Model.point_embed import PointEmbed
from occ_vae.Model.query_fusion import QueryFusion
from occ_vae.Model.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from occ_vae.Method.occ import make_occ_centers


def occ_to_pts(occ: torch.Tensor, centers: torch.Tensor):
    B, M, N, O = occ.shape

    coords_list = []
    batch_list = []

    for b in range(B):
        mask_b = occ[b] == 1  # 或者你要找的条件，比如 (occ[b] == 1) | (occ[b] == -1)
        idx = torch.where(mask_b)

        if idx[0].numel() == 0:
            continue

        # 用 idx 从 centers 取出对应点的坐标
        coords_b = centers[idx]  # (N_b, 3)

        coords_list.append(coords_b)
        batch_list.append(
            torch.full((coords_b.shape[0],), b, dtype=torch.long, device=occ.device)
        )

    coords = torch.cat(coords_list, dim=0)  # (N_total, 3)
    batch_idx = torch.cat(batch_list, dim=0)  # (N_total,)

    return coords, batch_idx


class TrilineVAE(nn.Module):
    def __init__(
        self,
        occ_size: int = 512,
        feat_num: int = 512,
        feat_dim: int = 32,
    ):
        super().__init__()
        self.occ_size = occ_size
        self.feat_num = feat_num
        self.feat_dim = feat_dim

        self.latent_dim = feat_dim + 1
        self.flatten_latent_dim = 3 * feat_num * self.latent_dim

        self.point_embed = PointEmbed(dim=self.latent_dim)

        self.ptv3_encoder = PointTransformerV3(self.latent_dim, enc_mode=True)

        self.triline_encoder = QueryFusion(512, self.feat_num, self.latent_dim * 3)

        self.mu_fc = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_fc = nn.Linear(self.latent_dim, self.latent_dim)

        self.delta_fc = nn.Sequential(
            nn.Linear(self.feat_num, self.feat_num - 1),
            nn.Sigmoid(),
        )

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.query_coords = make_occ_centers(occ_size)
        return

    def encode(
        self, occ: torch.Tensor, deterministic: bool = False
    ) -> Tuple[Triline, torch.Tensor]:
        if self.query_coords.device != occ.device:
            self.query_coords = self.query_coords.to(occ.device)

        coords, batch = occ_to_pts(occ, self.query_coords)

        coords_feature = self.point_embed(coords.unsqueeze(0)).squeeze(0)

        ptv3_data = {
            "coord": coords,
            "feat": coords_feature,
            "batch": batch,
            "grid_size": 1.0 / self.occ_size,
        }

        point = self.ptv3_encoder(ptv3_data)

        batch = point.batch
        feature = point.feat

        x = self.triline_encoder(feature, batch)

        x = x.view(x.shape[0], 3, self.feat_num, self.latent_dim)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mu, logvar, deterministic)

        x = posterior.sample()
        kl = posterior.kl()

        feats = x[..., :-1]
        deltas = x[..., -1]

        deltas = self.delta_fc(deltas)

        triline = Triline(feats, deltas)

        return triline, kl

    def decode(self, triline: Triline, coords: torch.Tensor) -> torch.Tensor:
        feat = triline.query(coords)
        feat = feat.reshape(feat.shape[:-2] + (feat.shape[-2] * feat.shape[-1],))
        occ_logits = self.decoder(feat).squeeze(-1)
        return occ_logits

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
        occ = data_dict["occ"]

        triline, kl = self.encode(occ)

        query_coords = self.query_coords.view(1, -1, 3).expand(occ.shape[0], -1, -1)

        logits = self.decodeLarge(triline, query_coords)

        result_dict = {
            "occ": logits,
            "kl": kl,
        }

        return result_dict
