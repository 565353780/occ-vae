import torch
import torch.nn.functional as F


def vae_loss(recon_logits, gt_occ, mu, logvar):
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_logits, gt_occ, reduction="mean"
    )
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl * 1.0
