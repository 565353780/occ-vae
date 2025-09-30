import sys

sys.path.append("../point-cept")

from triline_vae.Module.detector import Detector


def demo():
    model_file_path = "./output/v2/model_last.pth"
    use_ema = False
    batch_size = 1200000
    resolution = 128
    device = "cuda"

    detector = Detector(
        model_file_path,
        use_ema,
        batch_size,
        resolution,
        device,
    )

    mesh = detector.detectDataset(0)

    print(mesh)
    return True
