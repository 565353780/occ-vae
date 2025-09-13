import torch

from occ_vae.Model.sparse_conv_encoder import SparseConvEncoder
from occ_vae.Loss.vae import vae_loss
from occ_vae.Model.triline_vae import TrilineVAE


def test():
    input_dim = 64

    occ = torch.rand(1, 1, input_dim, input_dim, input_dim) > 0.5
    occ = occ.to(torch.float32).cuda()

    # occ: [B, 1, D, H, W]
    # coords: [B, Q, 3]  ∈ [-0.5, 0.5]^3
    # gt_occ: [B, Q] ∈ {0,1}

    model = TrilineVAE(input_dim).cuda()
    logits, mu, logvar = model(occ)
    loss = vae_loss(logits, occ, mu, logvar)
    loss.backward()

    print(occ.shape)
    print(logits.shape)
    print(mu.shape)
    print(logvar.shape)
    print(loss)

    """
    # 实例化并运行模型
    model = SparseConvEncoder().cuda()
    output = model(occ)
    print(output.shape)
    """
    return True
