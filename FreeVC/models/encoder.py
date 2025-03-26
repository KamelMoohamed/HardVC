import torch
from torch import nn

import commons

from .modules import WN


class Encoder(nn.Module):
    """
    Encoder module for feature extraction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Kernel size for convolution layers.
        dilation_rate (int): Dilation rate for convolutions.
        n_layers (int): Number of layers in the encoder.
        gin_channels (int, optional): Conditional input channels. Defaults to 0.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels

        # Initial projection layer
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Main encoder block using WaveNet-style architecture
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )

        # Projection layer to output means and log-variances
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def forward(self, x, x_lengths, g=None):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
            x_lengths (torch.Tensor): Lengths of each sequence in the batch.
            g (torch.Tensor, optional): Conditional input tensor. Defaults to None.

        Returns:
            tuple: Encoded latent representation `z`, mean `m`, log-variance `logs`, and mask `x_mask`.
        """
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask

        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
