import math
import torch
from torch import nn
from typing import Tuple

from triline_vae.Model.utils import PreNorm, Attention, FeedForward, subsample
from triline_vae.Model.utils import PointEmbed
from triline_vae.Model.bottleneck import NormalizedBottleneck
from triline_vae.Model.triline import Triline
from triline_vae.Model.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)
from triline_vae.Method.occ import make_occ_centers


def extract_fixed_size_pointcloud(occ, query_coords, target_N):
    """
    occ: [B, M, M, M] binary occupancy tensor
    query_coords: [B, M^3, 3] corresponding 3D coordinates
    target_N: int, number of points per batch to output

    Returns:
        points: [B, target_N, 3] tensor
    """
    B = occ.shape[0]
    occ_flat = occ.view(B, -1)  # [B, M^3]

    pointclouds = []

    for b in range(B):
        # Step 1: 获取当前 batch 中的占据点索引
        occ_indices = torch.nonzero(occ_flat[b] == 1).squeeze(1)  # [Nb]
        coords = query_coords[b][occ_indices]  # [Nb, 3]

        Nb = coords.shape[0]

        if Nb == 0:
            # 如果一个 batch 没有任何占据点，就用零点填充
            sampled = torch.zeros(
                (target_N, 3), device=occ.device, dtype=query_coords.dtype
            )
        elif Nb >= target_N:
            # 下采样
            rand_idx = torch.randperm(Nb, device=occ.device)[:target_N]
            sampled = coords[rand_idx]  # [N, 3]
        else:
            # 重复采样
            repeat_times = (target_N + Nb - 1) // Nb
            coords_repeated = coords.repeat((repeat_times, 1))  # [≥target_N, 3]
            sampled = coords_repeated[:target_N]  # [N, 3]

        pointclouds.append(sampled)

    # Stack into final tensor
    final_pointclouds = torch.stack(pointclouds, dim=0)  # [B, N, 3]
    return final_pointclouds


class VecSetAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        occ_size=512,
        feat_num=512,
        feat_dim=32,
        depth=24,
        latent_dim=768,
        output_dim=1,
        num_latents=1280,
        dim_head=64,
        bottleneck_dim: int = 1024,
        bottleneck_latent_dim: int = 32,
    ):
        super().__init__()
        self.feat_num = feat_num
        self.feat_dim = feat_dim

        dim = latent_dim + 1

        queries_dim = dim

        self.depth = depth

        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=dim // dim_head, dim_head=dim_head)
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )

        self.point_embed = PointEmbed(dim=dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(dim, heads=dim // dim_head, dim_head=dim_head),
                        ),
                        PreNorm(dim, FeedForward(dim)),
                    ]
                )
            )

        self.mean_fc = nn.Linear(dim, dim)
        self.logvar_fc = nn.Linear(dim, dim)

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=dim // dim_head, dim_head=dim_head),
        )

        self.to_outputs = nn.Sequential(
            nn.LayerNorm(queries_dim), nn.Linear(queries_dim, output_dim)
        )

        nn.init.zeros_(self.to_outputs[1].weight)
        nn.init.zeros_(self.to_outputs[1].bias)

        self.bottleneck = NormalizedBottleneck(
            dim=bottleneck_dim, latent_dim=bottleneck_latent_dim
        )

        self.query_coords = make_occ_centers(occ_size).reshape(1, -1, 3)
        return

    def encode(self, occ: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        sampled_pc = extract_fixed_size_pointcloud(occ, queries, self.num_latents)

        x = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context=pc_embeddings, mask=None) + x
        x = cross_ff(x) + x

        bottleneck = self.bottleneck.pre(x)
        return bottleneck

    def learn(self, x, deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.bottleneck.post(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar, deterministic)
        x = posterior.sample()
        kl = posterior.kl()
        return x, kl

    def decode(self, x, queries):
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        return self.to_outputs(latents)

    def forward(self, occ: torch.Tensor) -> dict:
        queries = self.query_coords.to(occ.device, dtype=occ.dtype).expand(
            occ.shape[0], -1, -1
        )

        bottleneck = self.encode(occ, queries)

        x, kl = self.learn(bottleneck["x"])

        if queries.shape[1] > 100000:
            N = 100000
            os = []
            for block_idx in range(math.ceil(queries.shape[1] / N)):
                o = self.decode(
                    x, queries[:, block_idx * N : (block_idx + 1) * N, :]
                ).squeeze(-1)
                os.append(o)
            o = torch.cat(os, dim=1)
        else:
            o = self.decode(x, queries).squeeze(-1)

        return {"o": o, **bottleneck}
