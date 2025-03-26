from typing import List, Tuple

import torch


def feature_loss(
    fmap_r: List[torch.Tensor], fmap_g: List[torch.Tensor]
) -> torch.Tensor:
    """
    Computes feature loss between real and generated feature maps.
    """
    loss = 0.0

    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            if rl.shape != gl.shape:  # Ensure shape match
                print(f"Shape mismatch: real {rl.shape}, generated {gl.shape}")
                min_shape = tuple(min(r, g) for r, g in zip(rl.shape, gl.shape))
                rl = rl[:, :, : min_shape[2]]  # Adjust shape
                gl = gl[:, :, : min_shape[2]]  # Adjust shape

            loss += torch.mean(torch.abs(rl.float().detach() - gl.float()))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """
    Computes discriminator loss for real and generated outputs.
    """
    r_losses, g_losses = [], []
    loss = sum(
        (r_loss := torch.mean((1 - dr.float()) ** 2))
        + (g_loss := torch.mean(dg.float() ** 2))
        or (r_losses.append(r_loss.item()), g_losses.append(g_loss.item()))[-1]
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs)
    )
    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Computes generator loss.
    """
    gen_losses = [torch.mean((1 - dg.float()) ** 2) for dg in disc_outputs]
    return sum(gen_losses), gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the KL divergence loss.

    Args:
        z_p, logs_q: Tensors of shape [b, h, t_t] (posterior sample and log variance)
        m_p, logs_p: Tensors of shape [b, h, t_t] (prior mean and log variance)
        z_mask: Mask tensor of shape [b, h, t_t]
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)
