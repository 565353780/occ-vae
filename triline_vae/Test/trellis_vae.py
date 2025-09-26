import sys

sys.path.append("../point-cept")

import torch

from triline_vae.Model.triline_vae import TrilineVAE


def test():
    occ_size = 128
    feat_num = 64
    feat_dim = 32

    triline_vae = TrilineVAE(occ_size, feat_num, feat_dim).cuda()

    occ = (
        (torch.randn([2, occ_size, occ_size, occ_size]) > 0.99).to(torch.float32).cuda()
    )

    data_dict = {
        "occ": occ,
    }

    result_dict = triline_vae(data_dict)

    pred_occ = result_dict["occ"]
    pred_kl = result_dict["kl"]

    print(occ.reshape(occ.shape[0], -1).shape)
    print(pred_occ.shape)
    print(pred_kl.shape)
    return True
