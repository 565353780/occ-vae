import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset

from octree_shape.Module.octree_builder import OctreeBuilder


class ShapeCodeDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        shape_code_folder_name: str,
        occ_depth: int,
        split: str = "train",
        dtype=torch.float32,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.occ_depth = occ_depth
        self.split = split
        self.dtype = dtype

        self.shape_code_folder_path = (
            self.dataset_root_folder_path + shape_code_folder_name + "/"
        )

        assert os.path.exists(self.shape_code_folder_path)

        self.output_error = False

        self.invalid_shape_code_file_path_list = []

        self.paths_list = []

        print("[INFO][ShapeCodeDataset::__init__]")
        print("\t start load shape_code datasets...")
        for root, _, files in os.walk(self.shape_code_folder_path):
            for file in files:
                file_type = file.split(".")[-1]
                if file_type not in ["npy"]:
                    continue

                if file.endswith("_tmp." + file_type):
                    continue

                shape_code_file_path = root + "/" + file
                self.paths_list.append(shape_code_file_path)
        return

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        shape_code_file_path = self.paths_list[index]

        if shape_code_file_path in self.invalid_shape_code_file_path_list:
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        try:
            octree_builder = OctreeBuilder()
            octree_builder.loadShapeCodeFile(shape_code_file_path)
        except KeyboardInterrupt:
            print("[INFO][ShapeCodeDataset::__getitem__]")
            print("\t stopped by the user (Ctrl+C).")
            exit()
        except Exception as e:
            if self.output_error:
                print("[ERROR][ShapeCodeDataset::__getitem__]")
                print("\t this shape_code file is not valid!")
                print("\t shape_code_file_path:", shape_code_file_path)
                print("\t error info:", e)

            self.invalid_shape_code_file_path_list.append(shape_code_file_path)
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        occ = octree_builder.getDepthOcc(self.occ_depth)

        data = {
            "occ": torch.from_numpy(occ).float(),
        }

        return data
