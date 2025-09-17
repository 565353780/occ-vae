import torch

from occ_vae.Model.triline_vae import TrilineVAE


def test():
    input_dim = 128

    occ = torch.rand(2, input_dim, input_dim, input_dim) > 0.5
    occ = occ.to(torch.float32).cuda()

    model = TrilineVAE(input_dim).cuda()
    logits, kl = model(occ)

    loss = torch.nn.BCEWithLogitsLoss()(occ.reshape(occ.shape[0], -1), logits)
    loss.backward()

    print(occ.shape)
    print(logits.shape)
    print(kl.shape)
    print(kl)
    print(loss)

    """
    # 实例化并运行模型
    model = SparseConvEncoder().cuda()
    output = model(occ)
    print(output.shape)
    """
    return True
