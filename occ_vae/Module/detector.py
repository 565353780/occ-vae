import os
import torch
from typing import Union

from octree_shape.Module.octree_builder import OctreeBuilder
from octree_shape.Method.occ import toOccCenters
from octree_shape.Method.render import renderBoxCentersMesh

from occ_vae.Model.triline import Triline
from occ_vae.Model.triline_vae import TrilineVAE
from occ_vae.Method.occ import make_occ_centers


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        use_ema: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
    ) -> None:
        self.use_ema = use_ema
        self.device = device
        self.dtype = dtype

        self.depth_max = 7
        self.feat_num = 64
        self.feat_dim = 32

        occ_size = 2**self.depth_max

        self.query_coords = (
            make_occ_centers(occ_size).view(1, -1, 3).to(self.device, dtype=self.dtype)
        )

        self.model = TrilineVAE(
            occ_size=occ_size,
            feat_num=self.feat_num,
            feat_dim=self.feat_dim,
        ).to(self.device, dtype=self.dtype)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model_file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_dict = torch.load(
            model_file_path, map_location=torch.device(self.device), weights_only=False
        )

        if self.use_ema:
            self.model.load_state_dict(model_dict["ema_model"])
        else:
            self.model.load_state_dict(model_dict["model"])

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def encodeMeshFile(self, mesh_file_path: str) -> Triline:
        self.model.eval()

        focus_center = [0, 0, 0.0]
        focus_length = 1.0
        normalize_scale = 0.99
        output_info = True

        octree_builder = OctreeBuilder(
            mesh_file_path,
            self.depth_max,
            focus_center,
            focus_length,
            normalize_scale,
            output_info,
        )

        occ = octree_builder.getDepthOcc(self.depth_max)

        gt_occ = torch.from_numpy(occ).unsqueeze(0).to(self.device, dtype=self.dtype)

        triline, _ = self.model.encode(gt_occ, True)

        pred_occ = self.model.decodeLarge(triline, self.query_coords)

        gt_occ = gt_occ.reshape(-1)
        pred_occ = pred_occ.reshape(-1)

        positive_occ_idxs = torch.where(gt_occ == 1)
        zero_occ_idxs = torch.where(gt_occ == 0)

        positive_pred_occ = pred_occ[positive_occ_idxs]

        zero_pred_occ = pred_occ[zero_occ_idxs]

        positive_acc = (
            positive_pred_occ > 0.5
        ).sum().item() / positive_pred_occ.numel()
        zero_acc = (zero_pred_occ < 0.5).sum().item() / zero_pred_occ.numel()

        print("[INFO][Detector::encodeMeshFile]")
        print("\t accuracy: positive", positive_acc, ", zero", zero_acc)
        return triline

    def renderTriline(self, triline: Triline) -> bool:
        pred_occ = self.model.decodeLarge(triline, self.query_coords)
        centers = toOccCenters(pred_occ)
        length = 0.5**self.depth_max
        renderBoxCentersMesh(centers, length)
        return True
