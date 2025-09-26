import torch
from torch import nn
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from triline_vae.Dataset.tsdf import TSDFDataset
from triline_vae.Model.triline_vae_v2 import TrilineVAEV2


class Trainer(BaseTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        use_amp: bool = False,
        quick_test: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.occ_size = 128
        self.feat_num = 64
        self.feat_dim = 32

        self.gt_sample_added_to_logger = False

        self.loss_fn = nn.MSELoss()

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            weights_only,
            device,
            dtype,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_amp,
            quick_test,
        )
        return

    def createDatasets(self) -> bool:
        eval = True
        self.dataloader_dict["tsdf"] = {
            "dataset": TSDFDataset(
                self.dataset_root_folder_path,
                "Objaverse_82K/sharp_edge_sdf",
                split="train",
                n_supervision=[21384, 10000, 10000],
            ),
            "repeat_num": 1,
        }

        if eval:
            self.dataloader_dict["eval"] = {
                "dataset": TSDFDataset(
                    self.dataset_root_folder_path,
                    "Objaverse_82K/sharp_edge_sdf",
                    split="val",
                    n_supervision=[21384, 10000, 10000],
                ),
            }

        if "eval" in self.dataloader_dict.keys():
            self.dataloader_dict["eval"]["dataset"].paths_list = self.dataloader_dict[
                "eval"
            ]["dataset"].paths_list[:4]

        return True

    def createModel(self) -> bool:
        self.model = TrilineVAEV2().to(self.device, dtype=self.dtype)
        return True

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        lambda_sharp_logits = 2.0
        lambda_coarse_logits = 1.0
        lambda_kl = 0.001

        gt_tsdf = data_dict["tsdf"]
        pred_tsdf = result_dict["tsdf"]
        kl = result_dict["kl"]
        number_sharp = data_dict["number_sharp"][0]

        gt_sharp_tsdf = gt_tsdf[:, :number_sharp]
        gt_coarse_tsdf = gt_tsdf[:, number_sharp:]

        pred_sharp_tsdf = pred_tsdf[:, :number_sharp]
        pred_coarse_tsdf = pred_tsdf[:, number_sharp:]

        positive_sharp_tsdf_idxs = torch.where(gt_sharp_tsdf > 0)
        negative_sharp_tsdf_idxs = torch.where(gt_sharp_tsdf < 0)

        positive_gt_sharp_tsdf = gt_sharp_tsdf[positive_sharp_tsdf_idxs]
        positive_pred_sharp_tsdf = pred_sharp_tsdf[positive_sharp_tsdf_idxs]

        negative_gt_sharp_tsdf = gt_sharp_tsdf[negative_sharp_tsdf_idxs]
        negative_pred_sharp_tsdf = pred_sharp_tsdf[negative_sharp_tsdf_idxs]

        positive_coarse_tsdf_idxs = torch.where(gt_coarse_tsdf > 0)
        negative_coarse_tsdf_idxs = torch.where(gt_coarse_tsdf < 0)

        positive_gt_coarse_tsdf = gt_coarse_tsdf[positive_coarse_tsdf_idxs]
        positive_pred_coarse_tsdf = pred_coarse_tsdf[positive_coarse_tsdf_idxs]

        negative_gt_coarse_tsdf = gt_coarse_tsdf[negative_coarse_tsdf_idxs]
        negative_pred_coarse_tsdf = pred_coarse_tsdf[negative_coarse_tsdf_idxs]

        loss_positive_sharp_tsdf = self.loss_fn(
            positive_pred_sharp_tsdf, positive_gt_sharp_tsdf
        )
        loss_negative_sharp_tsdf = self.loss_fn(
            negative_pred_sharp_tsdf, negative_gt_sharp_tsdf
        )

        loss_positive_coarse_tsdf = self.loss_fn(
            positive_pred_coarse_tsdf, positive_gt_coarse_tsdf
        )
        loss_negative_coarse_tsdf = self.loss_fn(
            negative_pred_coarse_tsdf, negative_gt_coarse_tsdf
        )

        loss_kl = torch.mean(kl)

        loss = (
            lambda_sharp_logits * (loss_positive_sharp_tsdf + loss_negative_sharp_tsdf)
            + lambda_coarse_logits
            * (loss_positive_coarse_tsdf + loss_negative_coarse_tsdf)
            + lambda_kl * loss_kl
        )

        positive_sharp_acc = (
            positive_pred_sharp_tsdf > 0
        ).sum().item() / positive_pred_sharp_tsdf.numel()
        negative_sharp_acc = (
            negative_pred_sharp_tsdf < 0
        ).sum().item() / negative_pred_sharp_tsdf.numel()

        positive_coarse_acc = (
            positive_pred_coarse_tsdf > 0
        ).sum().item() / positive_pred_coarse_tsdf.numel()
        negative_coarse_acc = (
            negative_pred_coarse_tsdf < 0
        ).sum().item() / negative_pred_coarse_tsdf.numel()

        loss_dict = {
            "Loss": loss,
            "LossPositiveSharpTSDF": loss_positive_sharp_tsdf,
            "LossNegativeSharpTSDF": loss_negative_sharp_tsdf,
            "LossPositiveCoarseTSDF": loss_positive_coarse_tsdf,
            "LossNegativeCoarseTSDF": loss_negative_coarse_tsdf,
            "PositiveSharpTSDFAcc": positive_sharp_acc,
            "NegativeSharpTSDFAcc": negative_sharp_acc,
            "PositiveCoarseTSDFAcc": positive_coarse_acc,
            "NegativeCoarseTSDFAcc": negative_coarse_acc,
            "LossKL": loss_kl,
        }

        return loss_dict

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        if is_training:
            data_dict["split"] = "train"
        else:
            data_dict["split"] = "val"

        return data_dict

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        # FIXME: skip this since it will occur NCCL error
        return True

        dataset = self.dataloader_dict["dino"]["dataset"]

        model.eval()

        data_dict = dataset.__getitem__(1)

        print("[INFO][BaseDiffusionTrainer::sampleModelStep]")
        print("\t start sample shape code....")

        if not self.gt_sample_added_to_logger:
            # render gt here

            # self.logger.addPointCloud("GT_MASH/gt_mash", pcd, self.step)

            self.gt_sample_added_to_logger = True

        # self.logger.addPointCloud(model_name + "/pcd_" + str(i), pcd, self.step)

        return True
