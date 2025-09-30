import os
import torch
import trimesh
from typing import Union

from triline_vae.Dataset.tsdf import TSDFDataset
from triline_vae.Model.triline_vae import TrilineVAE
from triline_vae.Model.triline_vae_v2 import TrilineVAEV2

from triline_vae.Method.tomesh import extractMesh


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        use_ema: bool = True,
        batch_size: int = 1200000,
        resolution: int = 128,
        device: str = "cpu",
    ) -> None:
        self.batch_size = batch_size
        self.resolution = resolution
        self.device = device

        self.model = TrilineVAEV2().to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path, use_ema)

        self.tsdf_dataset = TSDFDataset(
            "/home/chli/chLi/Dataset/",
            "Objaverse_82K/sharp_edge_sdf/",
            split="val",
            n_supervision=[21384, 10000, 10000],
        )
        return

    def loadModel(self, model_file_path: str, use_ema: bool = True) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path, map_location="cpu")

        if use_ema:
            model_state_dict = state_dict["ema_model"]
        else:
            model_state_dict = state_dict["model"]

        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def detect(
        self,
        coarse_surface: torch.Tensor,
        sharp_surface: torch.Tensor,
    ) -> Union[trimesh.Trimesh, None]:
        triline = self.model.encodeTriline(coarse_surface, sharp_surface)
        mesh = extractMesh(
            triline,
            self.model,
            self.resolution,
            self.batch_size,
            mode="odc",
        )
        return mesh

    @torch.no_grad()
    def detectDataset(self, data_idx: int) -> Union[trimesh.Trimesh, None]:
        data_dict = self.tsdf_dataset.__getitem__(data_idx)

        coarse_surface = data_dict["coarse_surface"].unsqueeze(0).to(self.device)
        sharp_surface = data_dict["sharp_surface"].unsqueeze(0).to(self.device)

        mesh = self.detect(coarse_surface, sharp_surface)
        return mesh
