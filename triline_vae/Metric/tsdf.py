import torch


@torch.no_grad()
def getTSDFAcc(
    gt_tsdf: torch.Tensor, pred_tsdf: torch.Tensor, dist_max: float
) -> float:
    """
    计算预测TSDF在有效距离范围内，且符号正确的准确率。

    Args:
        gt_tsdf: [B,...] 真实TSDF
        pred_tsdf: [B,...] 预测TSDF
        dist_max: float，距离阈值

    Returns:
        float: 预测符号与真实符号一致且距离有效范围内的点比例
    """
    # 找出真实TSDF有效范围索引
    valid_idxs = torch.where(torch.abs(gt_tsdf) <= dist_max)

    valid_gt = gt_tsdf[valid_idxs]
    valid_pred = pred_tsdf[valid_idxs]

    if valid_pred.numel() == 0:
        return 0.0

    # 判断符号是否一致 (大于等于0视为正， 小于0视为负)
    sign_match = (valid_gt >= 0) == (valid_pred >= 0)

    # 判断预测是否在有效距离范围内
    dist_match = torch.abs(valid_pred) <= dist_max

    # 两者同时满足才算正确
    correct = (sign_match & dist_match).sum().item()
    accuracy = correct / valid_pred.numel()

    return accuracy
