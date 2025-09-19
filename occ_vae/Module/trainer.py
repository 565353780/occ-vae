import torch
from torch import nn
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from occ_vae.Dataset.shape_code import ShapeCodeDataset
from occ_vae.Model.triline_vae import TrilineVAE


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

        self.loss_fn = nn.BCEWithLogitsLoss()

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
        self.dataloader_dict["dino"] = {
            "dataset": ShapeCodeDataset(
                self.dataset_root_folder_path,
                "Objaverse_82K/shape_code",
                occ_depth=7,
                split="train",
                dtype=self.dtype,
            ),
            "repeat_num": 1,
        }

        if eval:
            self.dataloader_dict["eval"] = {
                "dataset": ShapeCodeDataset(
                    self.dataset_root_folder_path,
                    "Objaverse_82K/shape_code",
                    occ_depth=7,
                    split="eval",
                    dtype=self.dtype,
                ),
            }

        if "eval" in self.dataloader_dict.keys():
            self.dataloader_dict["eval"]["dataset"].paths_list = self.dataloader_dict[
                "eval"
            ]["dataset"].paths_list[:4]

        return True

    def createModel(self) -> bool:
        self.model = TrilineVAE(
            occ_size=self.occ_size,
            feat_num=self.feat_num,
            feat_dim=self.feat_dim,
        ).to(self.device, dtype=self.dtype)
        return True

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        gt_occ = data_dict["occ"]
        pred_occ = result_dict["occ"]
        kl = result_dict["kl"]

        gt_occ = gt_occ.reshape(-1)
        pred_occ = pred_occ.reshape(-1)

        positive_occ_idxs = torch.where(gt_occ == 1)
        zero_occ_idxs = torch.where(gt_occ == 0)

        positive_gt_occ = gt_occ[positive_occ_idxs]
        positive_pred_occ = pred_occ[positive_occ_idxs]

        zero_gt_occ = gt_occ[zero_occ_idxs]
        zero_pred_occ = pred_occ[zero_occ_idxs]

        loss_positive_occ = self.loss_fn(positive_pred_occ, positive_gt_occ)
        loss_zero_occ = self.loss_fn(zero_pred_occ, zero_gt_occ)

        loss_kl = torch.mean(kl)

        loss = loss_positive_occ + loss_zero_occ + loss_kl

        positive_acc = (
            positive_pred_occ > 0.5
        ).sum().item() / positive_pred_occ.numel()
        zero_acc = (zero_pred_occ > 0.5).sum().item() / zero_pred_occ.numel()

        loss_dict = {
            "Loss": loss,
            "LossPositiveOcc": loss_positive_occ,
            "LossZeroOcc": loss_zero_occ,
            "PositiveAcc": positive_acc,
            "ZeroAcc": zero_acc,
            "LossKL": loss_kl,
        }

        return loss_dict

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        if is_training:
            data_dict["drop_prob"] = 0.0
        else:
            data_dict["drop_prob"] = 0.0

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
