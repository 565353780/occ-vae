import torch
from tqdm import trange

from occ_vae.Model.triline import Triline


def test():
    batch_size = 2
    feat_num = 512
    feat_dim = 32

    query_num = 16

    feats = torch.randn(
        [batch_size, 3, feat_num, feat_dim]
    )  # [B, 3, feat_num, feat_dim]

    triline = Triline(feats)

    for _ in trange(1000):
        coords = torch.rand(batch_size, query_num, 3) - 0.5

        feat = triline.query(coords)

    print(coords.shape)
    print(feat.shape)
    return True
