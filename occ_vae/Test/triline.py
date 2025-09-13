import torch
from tqdm import trange

from occ_vae.Model.triline import Triline


def test():
    batch_size = 2
    feat_num = 512
    feat_dim = 32

    query_occ_dim = 128

    feats = torch.randn(
        [batch_size, 3, feat_num, feat_dim]
    ).cuda()  # [B, 3, feat_num, feat_dim]

    triline = Triline(feats)

    for _ in trange(100):
        coords = (
            torch.rand(batch_size, query_occ_dim, query_occ_dim, query_occ_dim, 3) - 0.5
        ).cuda()

        feat = triline.query(coords)

    print(coords.shape)
    print(feat.shape)
    return True
