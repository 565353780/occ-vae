import torch

from occ_vae.Model.triline import Triline


def test():
    triline = Triline(N=64, C=32)

    coords = torch.rand(16, 3) - 0.5

    feat = triline(coords, mode="concat")

    print(coords.shape)
    print(feat.shape)
    return True
