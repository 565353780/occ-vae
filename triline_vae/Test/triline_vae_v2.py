from torch.utils.data import DataLoader

from triline_vae.Dataset.tsdf import TSDFDataset
from triline_vae.Model.triline_vae_v2 import TrilineVAEV2


def test():
    dataset_root_folder_path = "/home/chli/chLi/Dataset/Objaverse_82K/"
    dataset = TSDFDataset(
        dataset_root_folder_path,
        "sharp_edge_sdf",
        "train",
        [21384, 10000, 10000],
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

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
