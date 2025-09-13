import torch
import torch.nn.functional as F


class LearnableTriline:
    def __init__(self, x_deltas, x_feats, y_deltas, y_feats, z_deltas, z_feats):
        self.x_pos = self._normalize_deltas(x_deltas)  # [B, N]
        self.x_feat = x_feats  # [B, N, C]
        self.y_pos = self._normalize_deltas(y_deltas)
        self.y_feat = y_feats
        self.z_pos = self._normalize_deltas(z_deltas)
        self.z_feat = z_feats

    def _normalize_deltas(self, deltas):
        deltas = F.softplus(deltas)  # ensure > 0
        cum = torch.cumsum(deltas, dim=-1)  # [B, N-1]
        total = cum[:, -1:] + 1e-6
        pos = F.pad(cum / total, (1, 0), value=0.0)  # [B, N]
        return pos * 1.0 - 0.5  # scale to [-0.5, 0.5]

    def query(self, coords):  # coords: [B, Q, 3] ∈ [-0.5, 0.5]^3
        # coords: [B, Q, 3]
        def interp(pos, feat, c):
            idx = torch.searchsorted(pos, c, right=True).clamp(1, pos.shape[-1] - 1)
            left = idx - 1
            right = idx

            p0 = torch.gather(pos, 1, left)
            p1 = torch.gather(pos, 1, right)

            f0 = torch.gather(
                feat, 1, left.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
            )
            f1 = torch.gather(
                feat, 1, right.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
            )

            t = (c - p0) / (p1 - p0 + 1e-8)
            return (1 - t).unsqueeze(-1) * f0 + t.unsqueeze(-1) * f1

        x_feat = interp(self.x_pos, self.x_feat, coords[..., 0])
        y_feat = interp(self.y_pos, self.y_feat, coords[..., 1])
        z_feat = interp(self.z_pos, self.z_feat, coords[..., 2])

        return x_feat * y_feat * z_feat  # or concat and MLP
