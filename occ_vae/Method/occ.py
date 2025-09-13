import torch
from torchsparse import SparseTensor


def occ_to_torchsparse(occ: torch.Tensor) -> SparseTensor:
    coords = torch.nonzero(occ, as_tuple=False).int()

    feats = torch.ones((coords.shape[0], 1), dtype=torch.float32, device=occ.device)

    sparse_tensor = SparseTensor(coords=coords, feats=feats)
    return sparse_tensor
