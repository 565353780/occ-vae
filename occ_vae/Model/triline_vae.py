import torch
from torch import nn
from math import ceil
from torchsparse import SparseTensor
from torchsparse import nn as spnn

from occ_vae.Model.triline import Triline
from occ_vae.Model.point_embed import PointEmbed
from occ_vae.Model.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from occ_vae.Method.occ import make_occ_centers


def occ_to_torchsparse_tensor(occ, feature_dim=1):
    """
    occ: [B, M, M, M], 0/1 tensor
    returns:
        sparse_tensor: SparseTensor
        valid_batch_indices: list of int
    """
    B, M, _, _ = occ.shape
    coords_list = []
    feats_list = []
    valid_batch_indices = []

    for b in range(B):
        occ_coords = torch.nonzero(occ[b])  # [N_b, 3]
        if occ_coords.numel() == 0:
            continue
        # coords format: (b, z, y, x) — TorchSparse uses ZYX order
        batch_coords = torch.full(
            (occ_coords.size(0), 1), b, dtype=torch.int, device=occ.device
        )
        coords = torch.cat(
            [batch_coords, occ_coords[:, [2, 1, 0]].int()], dim=1
        )  # [N_b, 4]
        feats = torch.ones((occ_coords.size(0), feature_dim), dtype=torch.float32)

        coords_list.append(coords)
        feats_list.append(feats)
        valid_batch_indices.append(b)

    if len(coords_list) == 0:
        return None, []

    coords = torch.cat(coords_list, dim=0).contiguous().to(occ.device)
    feats = torch.cat(feats_list, dim=0).contiguous().to(occ.device)

    sparse_tensor = SparseTensor(feats, coords.int())
    return sparse_tensor, valid_batch_indices


def encode_occ_torchsparse(occ, encoder, out_dim=256):
    """
    occ: [B, M, M, M], 0/1 tensor
    encoder: TorchSparseEncoder
    returns: [B, D] global feature
    """
    B = occ.shape[0]
    device = occ.device
    features_padded = torch.zeros(B, out_dim, device=device)

    sparse_tensor, valid_batch_indices = occ_to_torchsparse_tensor(occ)
    if sparse_tensor is not None and len(valid_batch_indices) > 0:
        features = encoder(sparse_tensor)  # [B', D]
        for i, b in enumerate(valid_batch_indices):
            features_padded[b] = features[i]

    return features_padded  # [B, D]


class TorchSparseEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(1, 32, kernel_size=3, stride=1),
            spnn.ReLU(),
            spnn.Conv3d(32, 64, kernel_size=2, stride=2),
            spnn.ReLU(),
            spnn.Conv3d(64, 128, kernel_size=2, stride=2),
            spnn.ReLU(),
            spnn.GlobalAvgPool(),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: SparseTensor):
        x = self.net(x)  # Global pooled sparse tensor
        return self.fc(x)  # x: [B', 128] → [B', D]


class TrilineVAE(nn.Module):
    def __init__(
        self,
        occ_size=512,
        feat_num=512,
        feat_dim=32,
    ):
        super().__init__()
        self.feat_num = feat_num
        self.feat_dim = feat_dim

        self.latent_dim = feat_dim + 1
        self.flatten_latent_dim = 3 * feat_num * self.latent_dim

        self.encoder = TorchSparseEncoder(self.flatten_latent_dim)

        self.mu_fc = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_fc = nn.Linear(self.latent_dim, self.latent_dim)

        self.delta_fc = nn.Linear(self.feat_num, self.feat_num - 1)

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.query_coords = make_occ_centers(occ_size).reshape(1, -1, 3)
        return

    def encode(self, occ: torch.Tensor, deterministic: bool = False):
        x = encode_occ_torchsparse(occ, self.encoder, self.flatten_latent_dim)

        x = x.view(x.shape[0], 3, self.feat_num, self.latent_dim)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mu, logvar, deterministic)

        x = posterior.sample()
        kl = posterior.kl()
        return x, kl

    def decode(self, triline, coords):
        feat = triline.query(coords)
        feat = feat.reshape(feat.shape[:-2] + (feat.shape[-2] * feat.shape[-1],))
        occ_logits = self.decoder(feat).squeeze(-1)
        return occ_logits

    def forward(self, data_dict: dict) -> dict:
        occ = data_dict["occ"]

        x, kl = self.encode(occ)

        feats = x[..., :-1]
        deltas = x[..., -1]

        deltas = self.delta_fc(deltas)

        triline = Triline(feats, deltas)

        query_coords = self.query_coords.to(occ.device, dtype=occ.dtype).expand(
            occ.shape[0], -1, -1
        )

        N = 100000
        if query_coords.shape[1] > N:
            logits = []
            for block_idx in range(ceil(query_coords.shape[1] / N)):
                curr_logits = self.decode(
                    triline, query_coords[:, block_idx * N : (block_idx + 1) * N, :]
                )
                logits.append(curr_logits)
            logits = torch.cat(logits, dim=1)
        else:
            logits = self.decode(triline, query_coords)

        result_dict = {
            "occ": logits,
            "kl": kl,
        }

        return result_dict
