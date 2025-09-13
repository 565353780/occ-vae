import torch

from occ_vae.Model.sparse_conv_encoder import SparseConvEncoder


def test():
    input_dim = 2

    occ = torch.rand(2, input_dim, input_dim, input_dim) > 0.5
    occ = occ.cuda()

    # 实例化并运行模型
    model = SparseConvEncoder().cuda()
    output = model(occ)
    print(output.shape)
    return True
