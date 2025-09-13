import torch
from torch import nn

from occ_vae.Model.learnable_triline import LearnableTriline
from occ_vae.Method.occ import make_occ_centers


class TrilineVAE(nn.Module):
    def __init__(self, occ_size=512, latent_dim=128, feat_num=512, feat_dim=32):
        super().__init__()
        self.feat_num = feat_num
        self.feat_dim = feat_dim

        # 例如 512 → 256 → 128 → 64 → 32 → 16 → 8
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1),  # 512 → 256
            nn.ReLU(),
            nn.Conv3d(32, 64, 4, 2, 1),  # 256 → 128
            nn.ReLU(),
            nn.Conv3d(64, 128, 4, 2, 1),  # 128 → 64
            nn.ReLU(),
            nn.Conv3d(128, 256, 4, 2, 1),  # 64 → 32
            nn.ReLU(),
            nn.Conv3d(256, 512, 4, 2, 1),  # 32 → 16
            nn.ReLU(),
            nn.Conv3d(512, 512, 4, 2, 1),  # 16 → 8
            nn.ReLU(),
        )

        dummy_input = torch.zeros(1, 1, occ_size, occ_size, occ_size)
        with torch.no_grad():
            out = self.encoder(dummy_input)
            self.flat_dim = out.view(1, -1).shape[1]  # B × C × D × H × W → [1, C*D*H*W]
            print("[INFO][TrilineVAE::__init__]")
            print("\t flat dim:", self.flat_dim)

        self.fc = nn.Linear(self.flat_dim, latent_dim * 2)

        # MLPs to map z → deltas + features
        self.mlp_deltas = nn.Sequential(nn.Linear(latent_dim, 3 * (feat_num - 1)))
        self.mlp_feats = nn.Sequential(
            nn.Linear(latent_dim, 3 * feat_num * feat_dim), nn.ReLU()
        )

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.query_coords = make_occ_centers(occ_size)
        return

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # [B, C*D*H*W]
        stats = self.fc(x)
        mu, logvar = stats.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, triline, coords):  # coords: [B, Q, 3] ∈ [-0.5, 0.5]^3
        feat = triline.query(coords)  # [B, Q, feat_dim]
        occ_logits = self.decoder(feat).squeeze(-1)  # [B, Q]
        return occ_logits

    def forward(self, occ):
        mu, logvar = self.encode(occ)
        z = self.reparameterize(mu, logvar)

        # Predict deltas and features
        deltas = self.mlp_deltas(z).view(-1, 3, self.feat_num - 1)
        feats = self.mlp_feats(z).view(-1, 3, self.feat_num, self.feat_dim)

        triline = LearnableTriline(
            deltas[:, 0],
            feats[:, 0],
            deltas[:, 1],
            feats[:, 1],
            deltas[:, 2],
            feats[:, 2],
        )

        query_coords = (
            self.query_coords.to(occ.device, dtype=occ.dtype)
            .unsqueeze(0)
            .expand(occ.shape[0], -1, -1)
        )

        logits = self.decode(triline, query_coords)

        pred_occ = logits.reshape_as(occ)

        return pred_occ, mu, logvar
