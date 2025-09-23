import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class SDFDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        split: str = "train",
        n_qry: int = 200,
        noise_label: str = "0_25",
        train_percent: float = 0.99,
        near_surface_dist: float = 1.0 / 512.0,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split
        self.n_qry = n_qry
        self.near_surface_dist = near_surface_dist

        self.mash_folder_path = (
            self.dataset_root_folder_path + "Objaverse_82K/manifold_mash/"
        )
        assert os.path.exists(self.mash_folder_path)

        self.sdf_folder_path = (
            self.dataset_root_folder_path
            + "Objaverse_82K/manifold_sdf_"
            + noise_label
            + "/"
        )
        assert os.path.exists(self.sdf_folder_path)

        self.paths_list = []

        print("[INFO][SDFDataset::__init__]")
        print("\t start load dataset:", self.mash_folder_path)
        for root, _, files in os.walk(self.mash_folder_path):
            for file in files:
                if not file.endswith(".npy"):
                    continue

                rel_file_basepath = (
                    os.path.relpath(root, self.mash_folder_path) + "/" + file[:-4]
                )

                mash_file_path = self.mash_folder_path + rel_file_basepath + ".npy"
                assert os.path.exists(mash_file_path)

                sdf_file_path = self.sdf_folder_path + rel_file_basepath + ".npy"
                if not os.path.exists(sdf_file_path):
                    continue

                self.paths_list.append([mash_file_path, sdf_file_path])

        self.paths_list.sort(key=lambda x: x[0])

        train_data_num = max(int(len(self.paths_list) * train_percent), 1)

        if self.split == "train":
            self.paths_list = self.paths_list[:train_data_num]
        else:
            self.paths_list = self.paths_list[train_data_num:]
        return

    def __len__(self):
        return len(self.paths_list)

    def getRandomItem(self):
        random_idx = random.randint(0, len(self.paths_list) - 1)
        return self.__getitem__(random_idx)

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        mash_file_path, sdf_file_path = self.paths_list[index]

        try:
            sdf_data = np.load(sdf_file_path)
        except:
            return self.getRandomItem()

        if np.isnan(sdf_data).any() or np.isinf(sdf_data).any():
            return self.getRandomItem()

        mash_params = loadMashTensor(mash_file_path)
        if mash_params is None:
            return self.getRandomItem()

        permute_idxs = np.random.permutation(mash_params.shape[0])

        mash_params = mash_params[permute_idxs]

        qry = sdf_data[:, :3]
        sdf = sdf_data[:, 3]

        positive_sdf_idxs = np.where(sdf <= 0.0)[0]
        negative_sdf_idxs = np.where(sdf > 0.0)[0]

        occ = np.ones_like(sdf)
        occ[negative_sdf_idxs] = 0.0

        near_surface_sdf_mask = (sdf < self.near_surface_dist) & (
            sdf > -self.near_surface_dist
        )
        occ[near_surface_sdf_mask] = (
            0.5 - 0.5 / self.near_surface_dist * sdf[near_surface_sdf_mask]
        )

        positive_sdf_num = self.n_qry // 2

        if positive_sdf_num > positive_sdf_idxs.shape[0]:
            positive_sdf_num = positive_sdf_idxs.shape[0]

        negative_sdf_num = self.n_qry - positive_sdf_num

        if negative_sdf_num > negative_sdf_idxs.shape[0]:
            negative_sdf_num = negative_sdf_idxs.shape[0]

        positive_idxs = np.random.choice(positive_sdf_idxs, positive_sdf_num)
        negative_idxs = np.random.choice(negative_sdf_idxs, negative_sdf_num)

        idxs = np.hstack([positive_idxs, negative_idxs])

        perm = idxs[np.random.permutation(positive_sdf_num + negative_sdf_num)]

        qry = qry[perm]
        occ = occ[perm]

        feed_dict = {
            "mash_params": mash_params,
            "qry": torch.tensor(qry).float(),
            "occ": torch.tensor(occ).float(),
        }

        return feed_dict
