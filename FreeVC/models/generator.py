import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm

from commons import init_weights
from models.modules import ResBlock1, ResBlock2


class Generator(nn.Module):
    """
    A waveform generator using transposed convolutions and residual blocks.

    Args:
        initial_channel (int): Number of input channels.
        resblock (nn.Module): Residual block type.
        resblock_kernel_sizes (list): Kernel sizes for residual blocks.
        resblock_dilation_sizes (list): Dilation sizes for residual blocks.
        upsample_rates (list): Upsampling factors.
        upsample_initial_channel (int): Initial channel size for upsampling.
        upsample_kernel_sizes (list): Kernel sizes for upsampling layers.
        gin_channels (int, optional): Number of channels for conditional input. Defaults to 0.
    """

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super().__init__()

        # Number of upsampling layers
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        # Pre-convolution layer before upsampling
        self.conv_pre = Conv1d(
            initial_channel,
            upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # Upsampling layers
        self.ups = nn.ModuleList(
            [
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2,
                    )
                )
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
            ]
        )

        # Residual blocks
        self.resblocks = nn.ModuleList(
            [
                resblock(upsample_initial_channel // (2 ** (i + 1)), k, d)
                for i in range(self.num_upsamples)
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        # Post-convolution layer
        self.conv_post = Conv1d(
            upsample_initial_channel // (2**self.num_upsamples),
            1,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )

        # Apply weight initialization
        self.ups.apply(init_weights)

        # Conditional input handling
        self.cond = (
            nn.Conv1d(gin_channels, upsample_initial_channel, kernel_size=1)
            if gin_channels
            else None
        )

    def forward(self, x, g=None):
        """
        Forward pass for waveform generation.

        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, time_steps).
            g (torch.Tensor, optional): Conditional input (batch_size, cond_channels, time_steps). Defaults to None.

        Returns:
            torch.Tensor: Generated waveform.
        """
        x = self.conv_pre(x)

        # Apply conditional input
        if g is not None and self.cond:
            x = x + self.cond(g)

        # Upsampling and residual blocks
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.2)
            x = self.ups[i](x)
            x = (
                sum(
                    self.resblocks[i * self.num_kernels + j](x)
                    for j in range(self.num_kernels)
                )
                / self.num_kernels
            )

        # Final activation and output
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_post(x)
        return torch.tanh(x)
