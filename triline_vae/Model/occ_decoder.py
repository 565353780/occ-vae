import torch
import torch.nn as nn
import torch.nn.functional as F


class OccDecoder(nn.Module):
    def __init__(
        self,
        feat_dim=32,
        hidden_dim=128,
        num_layers=5,
        use_xyz=True,
        use_posenc=False,
        posenc_freq=10,
    ):
        super().__init__()
        self.use_xyz = use_xyz
        self.use_posenc = use_posenc
        input_dim = feat_dim * 3

        if use_xyz:
            if use_posenc:
                input_dim += posenc_freq * 2 * 3  # sin/cos for each dim
            else:
                input_dim += 3  # raw xyz

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, feat, coords=None):
        """
        feat: (B, N, feat_dim * 3) - triline interpolated feature
        coords: (B, N, 3) - xyz coords (optional)
        """
        x = feat

        if self.use_xyz and coords is not None:
            if self.use_posenc:
                coords = self.positional_encoding(coords)  # (B, N, posenc_dim)
            x = torch.cat([x, coords], dim=-1)

        x = F.relu(self.fc_in(x))
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection

        occ = self.fc_out(x)
        return occ

    def positional_encoding(self, coords, L=10):
        """Sinusoidal positional encoding"""
        freq_bands = 2.0 ** torch.arange(L, device=coords.device)  # (L,)
        coords = coords.unsqueeze(-1) * freq_bands  # (B, N, 3, L)
        coords = coords.view(*coords.shape[:2], -1)  # (B, N, 3*L)
        sin = torch.sin(coords)
        cos = torch.cos(coords)
        return torch.cat([sin, cos], dim=-1)  # (B, N, 3 * 2 * L)
