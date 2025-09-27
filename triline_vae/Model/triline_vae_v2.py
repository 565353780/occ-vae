import math
import torch
import torch.nn as nn

from triline_vae.Model.triline import Triline
from triline_vae.Model.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)
from triline_vae.Model.Layer.fourier_embedder import FourierEmbedder
from triline_vae.Model.transformer.perceiver_1d import Perceiver
from triline_vae.Model.occ_decoder import OccDecoder
from triline_vae.Model.tsdf_decoder import TSDFDecoder
from triline_vae.Model.perceiver_cross_attention_encoder import (
    PerceiverCrossAttentionEncoder,
)


class TrilineVAEV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.feat_dim: int = 256

        self.use_downsample: bool = True
        self.num_latents: int = 256
        self.embed_dim: int = 64
        self.width: int = 3 * self.feat_dim
        self.point_feats: int = 3
        self.embed_point_feats: bool = False
        self.out_dim: int = 1
        self.num_freqs: int = 8
        self.include_pi: bool = False
        self.heads: int = 12
        self.num_encoder_layers: int = 8
        self.num_decoder_layers: int = 16
        self.init_scale: float = 0.25
        self.qkv_bias: bool = False
        self.use_ln_post: bool = True
        self.use_flash: bool = True

        self.embedder = FourierEmbedder(
            num_freqs=self.num_freqs,
            include_pi=self.include_pi,
        )
        self.init_scale = self.init_scale * math.sqrt(1.0 / self.width)
        self.encoder = PerceiverCrossAttentionEncoder(
            use_downsample=self.use_downsample,
            embedder=self.embedder,
            num_latents=self.num_latents,
            point_feats=self.point_feats,
            embed_point_feats=self.embed_point_feats,
            width=self.width,
            heads=self.heads,
            layers=self.num_encoder_layers,
            init_scale=self.init_scale,
            qkv_bias=self.qkv_bias,
            use_ln_post=self.use_ln_post,
            use_flash=self.use_flash,
        )

        self.pre_kl = nn.Linear(self.width, self.embed_dim * 2)
        self.post_kl = nn.Linear(self.embed_dim, self.width)

        self.transformer = Perceiver(
            n_ctx=self.num_latents,
            width=self.width,
            layers=self.num_decoder_layers,
            heads=self.heads,
            init_scale=self.init_scale,
            qkv_bias=self.qkv_bias,
            use_flash=self.use_flash,
        )

        self.decoder = TSDFDecoder(
            in_dim=3 * self.feat_dim,
            hidden_dim=self.feat_dim,
        )
        return

    def encode(
        self,
        coarse_surface: torch.FloatTensor,
        sharp_surface: torch.FloatTensor,
    ):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """

        coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:]
        sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:]
        shape_latents = self.encoder(
            coarse_pc, sharp_pc, coarse_feats, sharp_feats, split=self.split
        )
        return shape_latents

    def encode_kl_embed(
        self, latents: torch.FloatTensor, sample_posterior: bool = True
    ):
        posterior = None
        moments = self.pre_kl(latents)  # 103，256，768 -》 103，256，128
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        if sample_posterior:
            kl_embed = posterior.sample()  # 1，768，64
        else:
            kl_embed = posterior.mode()

        kl = posterior.kl()
        return kl_embed, kl

    def decode(self, latents: torch.FloatTensor) -> Triline:
        """
        Args:
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            triline (Triline)
        """
        latents = self.post_kl(
            latents
        )  # [B, num_latents, embed_dim] -> [B, num_latents, width]

        latents = self.transformer(latents)

        latents_chunks = torch.chunk(
            latents, 3, dim=-1
        )  # list of 3 tensors, each [B, N, C/3]
        latents = torch.stack(latents_chunks, dim=1)

        triline = Triline(latents)

        return triline

    def query(self, queries: torch.FloatTensor, triline: Triline):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            triline (Triline)

        Returns:
            logits (torch.FloatTensor): [B, N], tsdf logits
        """
        # (B, N, 3, feat_dim)
        feat = triline.query(queries)
        feat = feat.reshape(feat.shape[0], feat.shape[1], 3 * self.feat_dim)
        logits = self.decoder(feat).squeeze(-1)
        tsdf = nn.Sigmoid()(logits) * 2.0 - 1.0
        return tsdf

    def forward(
        self,
        data_dict: dict,
    ) -> dict:
        coarse_surface = data_dict["coarse_surface"]
        sharp_surface = data_dict["sharp_surface"]
        queries = data_dict["rand_points"]
        self.split = data_dict["split"]

        shape_latents = self.encode(coarse_surface, sharp_surface)

        kl_embed, kl = self.encode_kl_embed(shape_latents, sample_posterior=True)

        triline = self.decode(kl_embed)

        tsdf = self.query(queries, triline)

        result_dict = {
            "tsdf": tsdf,
            "kl": kl,
        }

        return result_dict
