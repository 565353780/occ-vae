import torch
import mcubes
import trimesh
import numpy as np
from torch import nn
from math import ceil
from tqdm import tqdm
from functools import partial

from triline_vae.Lib.ODC.occupancy_dual_contouring import occupancy_dual_contouring
from triline_vae.Model.triline import Triline


@torch.no_grad()
def toOCC(qry: torch.Tensor, triline: Triline, model: nn.Module) -> torch.Tensor:
    feat = triline.query(qry.to(torch.float32).unsqueeze(0))
    feat = feat.reshape(feat.shape[0], feat.shape[1], 3 * model.feat_dim)
    logits = model.decoder(feat).squeeze(-1)
    tsdf = nn.Sigmoid()(logits) * 2.0 - 1.0
    return tsdf.squeeze(0)


@torch.no_grad()
def extractMesh(
    triline: Triline,
    model: nn.Module,
    resolution: int = 128,
    batch_size: int = 1200000,
    mode: str = "odc",
) -> trimesh.Trimesh:
    assert mode in ["odc", "mc"]

    device = triline.feats.device

    if mode == "odc":
        odc = occupancy_dual_contouring(device)

        vertices, triangles = odc.extract_mesh(
            imp_func=partial(toOCC, triline=triline, model=model),
            min_coord=[-1.05, -1.05, -1.05],
            max_coord=[1.05, 1.05, 1.05],
            num_grid=resolution,
            isolevel=0.0,
            batch_size=batch_size,
            outside=True,
        )

        mesh = trimesh.Trimesh(vertices.cpu(), triangles.cpu())

        return mesh

    gap = 2.0 / resolution
    x = np.linspace(-1, 1, resolution + 1)
    y = np.linspace(-1, 1, resolution + 1)
    z = np.linspace(-1, 1, resolution + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = (
        torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32))
        .view(3, -1)
        .transpose(0, 1)[None]
        .to(device, non_blocking=True)
    )

    chunk_num = ceil(grid.shape[1] / batch_size)
    if device == torch.device("cpu"):
        chunk_num *= 10

    grids = torch.chunk(grid, chunk_num, dim=1)

    logits_list = []

    for chunk_grid in tqdm(grids):
        chunk_logits = toOCC(chunk_grid, triline, model)
        logits_list.append(chunk_logits)

    logits = torch.hstack(logits_list)

    logits = logits.detach()

    volume = (
        logits.view(resolution + 1, resolution + 1, resolution + 1)
        .permute(1, 0, 2)
        .cpu()
        .numpy()
    )
    verts, faces = mcubes.marching_cubes(volume, 0)

    verts *= gap
    verts -= 1

    mesh = trimesh.Trimesh(verts, faces)

    return mesh
