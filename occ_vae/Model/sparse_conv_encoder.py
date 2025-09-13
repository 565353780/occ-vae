import torch
import torchsparse.nn as spnn
from torch import nn

from occ_vae.Method.occ import occ_to_torchsparse


class SparseConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = spnn.Conv3d(1, 16, kernel_size=3, stride=1)
        self.conv2 = spnn.Conv3d(16, 32, kernel_size=3, stride=1)
        self.fc = nn.Linear(32, 10)
        return

    def forward(self, occ: torch.Tensor) -> torch.Tensor:
        x = occ_to_torchsparse(occ)

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.feats.view(x.feats.size(0), -1)
        x = self.fc(x)
        print(x)
        print(x.dtype)
        print(x.device)
        print(x.shape)
        return x
