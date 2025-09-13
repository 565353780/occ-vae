import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset

from octree_shape.Module.occ_sampler import OccSampler


class OccDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        mesh_folder_name: str,
        subdiv_depth: int,
        occ_depth: int,
        split: str = "train",
        dtype=torch.float32,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.subdiv_depth = subdiv_depth
        self.occ_depth = occ_depth
        self.split = split
        self.dtype = dtype

        self.mesh_folder_path = self.dataset_root_folder_path + mesh_folder_name + "/"

        assert os.path.exists(self.mesh_folder_path)

        self.output_error = False

        self.invalid_mesh_file_path_list = []

        self.paths_list = []

        print("[INFO][OccDataset::__init__]")
        print("\t start load mesh datasets...")
        for root, _, files in os.walk(self.mesh_folder_path):
            for file in files:
                file_type = file.split(".")[-1]
                if file_type not in ["obj", "ply", "glb"]:
                    continue

                if file.endswith("_tmp." + file_type):
                    continue

                mesh_file_path = root + "/" + file
                self.paths_list.append(mesh_file_path)

        self.query_idxs = np.zeros(len(self.paths_list), dtype=int)
        return

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        mesh_file_path = self.paths_list[index]

        if mesh_file_path in self.invalid_mesh_file_path_list:
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        try:
            occ_sampler = OccSampler(
                mesh_file_path,
                self.subdiv_depth,
                self.occ_depth,
                output_info=False,
            )
        except KeyboardInterrupt:
            print("[INFO][OccDataset::__getitem__]")
            print("\t stopped by the user (Ctrl+C).")
            exit()
        except Exception as e:
            if self.output_error:
                print("[ERROR][OccDataset::__getitem__]")
                print("\t this mesh file is not valid!")
                print("\t mesh_file_path:", mesh_file_path)
                print("\t error info:", e)

            self.invalid_mesh_file_path_list.append(mesh_file_path)
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        occ = occ_sampler.queryOrderedOcc(self.query_idxs[index])
        self.query_idxs[index] += 1

        data = {
            "occ": occ,
        }

        return data
