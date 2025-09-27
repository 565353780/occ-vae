import torch


@torch.no_grad()
def getTSDFAcc(
    gt_tsdf: torch.Tensor,
    pred_tsdf: torch.Tensor,
    dist_max: float,
) -> float:
    """
    向量化计算预测TSDF正确率。
    对于截断边界 ±1 的点，只要预测超出边界且符号一致也算正确。
    对于其他点，要求 |pred - gt| <= dist_max 且符号一致。

    Args:
        gt_tsdf: [B,...] 真实TSDF，截断在 [-1, 1]
        pred_tsdf: [B,...] 预测TSDF
        dist_max: float，非截断点的误差阈值

    Returns:
        float: 正确点比例
    """
    # 只考虑 GT 在 [-1, 1] 的有效点
    valid_mask = (gt_tsdf >= -1.0) & (gt_tsdf <= 1.0)
    gt = gt_tsdf[valid_mask]
    pred = pred_tsdf[valid_mask]

    if gt.numel() == 0:
        return 0.0

    # 符号一致
    sign_match = (gt >= 0) == (pred >= 0)

    # 对截断 ±1 点特殊处理
    at_pos1 = gt >= 1.0
    at_neg1 = gt <= -1.0
    at_mid = (~at_pos1) & (~at_neg1)

    # 正确条件
    correct = torch.zeros_like(gt, dtype=torch.bool)
    # 中间点：误差在阈值内且符号一致
    correct[at_mid] = (torch.abs(pred[at_mid] - gt[at_mid]) <= dist_max) & sign_match[
        at_mid
    ]
    # 正截断点：预测 >= 1 且符号一致
    correct[at_pos1] = (pred[at_pos1] >= 1.0) & sign_match[at_pos1]
    # 负截断点：预测 <= -1 且符号一致
    correct[at_neg1] = (pred[at_neg1] <= -1.0) & sign_match[at_neg1]

    accuracy = correct.sum().item() / gt.numel()
    return accuracy
