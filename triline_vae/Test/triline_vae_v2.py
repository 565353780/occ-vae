import numpy as np
import open3d as o3d
from typing import Union
from torch.utils.data import DataLoader

from triline_vae.Dataset.tsdf import TSDFDataset
from triline_vae.Model.triline_vae_v2 import TrilineVAEV2


def getPcd(
    points: np.ndarray,
    normals: Union[np.ndarray, None] = None,
    colors: Union[np.ndarray, list] = [1.0, 0.0, 0.0],
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    pcd.colors = o3d.utility.Vector3dVector(np.tile(colors, (points.shape[0], 1)))
    return pcd


def viewData(data):
    fps_coarse_surface = data["coarse_surface"]
    fps_sharp_surface = data["sharp_surface"]
    rand_points = data["rand_points"]

    fps_sharp_surface_pcd = getPcd(
        fps_sharp_surface[:, :3],
        fps_sharp_surface[:, 3:],
    )

    fps_coarse_surface_pcd = getPcd(
        fps_coarse_surface[:, :3],
        fps_coarse_surface[:, 3:],
    )

    rand_points_pcd = getPcd(rand_points[:, :3])

    fps_sharp_surface_pcd.translate([2.5, 0, 0])

    fps_coarse_surface_pcd.translate([7.5, 0, 0])

    rand_points_pcd.translate([10.0, 0, 0])

    o3d.visualization.draw_geometries(
        [
            fps_sharp_surface_pcd,
            fps_coarse_surface_pcd,
            rand_points_pcd,
        ]
    )
    return True


def test():
    dataset_root_folder_path = "/home/chli/chLi/Dataset/Objaverse_82K/"
    vis_dataset = False

    dataset = TSDFDataset(
        dataset_root_folder_path,
        "sharp_edge_sdf",
        "train",
        [21384, 10000, 10000],
    )

    if vis_dataset:
        for i in range(len(dataset)):
            data = dataset[i]
            viewData(data)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = TrilineVAEV2().cuda()

    for data in dataloader:
        for key, item in data.items():
            try:
                data[key] = data[key].cuda()
            except:
                continue
        data["split"] = "train"

        result = model(data)

        for key, item in result.items():
            print(key, ":", item.shape)
            break

    return True
