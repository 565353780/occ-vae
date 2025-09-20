import os
import torch

from occ_vae.Module.detector import Detector


def demo():
    model_file_path = "./output/v2/model_last.pth"
    use_ema = False
    device = "cpu"
    dtype = torch.float32

    mesh_file_path = os.environ["HOME"] + "/chLi/Dataset/vae-eval/mesh/002.obj"

    detector = Detector(
        model_file_path,
        use_ema,
        device,
        dtype,
    )

    triline = detector.encodeMeshFile(mesh_file_path)

    detector.renderTriline(triline)
    return True
