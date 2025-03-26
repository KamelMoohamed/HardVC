import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import spectral_norm, weight_norm

from commons import get_padding


class DiscriminatorP(nn.Module):
    """A periodic discriminator that analyzes periodic patterns in generated audio."""

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_channels, out_channels in zip(
                    [1, 32, 128, 512, 1024], [32, 128, 512, 1024, 1024]
                )
            ]
        )

        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """Processes the input tensor and returns a prediction with feature maps."""
        fmap = []
        b, c, t = x.shape

        if t % self.period != 0:  # Padding for consistency
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")

        x = x.view(
            b, c, -1, self.period
        )  # Reshape to (batch, channels, time // period, period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), negative_slope=0.2)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1)

        return x, fmap


class DiscriminatorS(nn.Module):
    """A spectral discriminator that evaluates the overall waveform quality."""

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv1d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding=padding,
                        groups=groups,
                    )
                )
                for in_channels, out_channels, kernel, stride, groups, padding in [
                    (1, 16, 15, 1, 1, 7),
                    (16, 64, 41, 4, 4, 20),
                    (64, 256, 41, 4, 16, 20),
                    (256, 1024, 41, 4, 64, 20),
                    (1024, 1024, 41, 4, 256, 20),
                    (1024, 1024, 5, 1, 1, 2),
                ]
            ]
        )

        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """Processes the input tensor and returns a prediction with feature maps."""
        fmap = []

        for conv in self.convs:
            x = F.leaky_relu(conv(x), negative_slope=0.2)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """A multi-period discriminator that combines spectral and periodic discriminators."""

    PERIODS = [2, 3, 5, 7, 11]

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm)]
            + [
                DiscriminatorP(period, use_spectral_norm=use_spectral_norm)
                for period in self.PERIODS
            ]
        )

    def forward(self, y, y_hat):
        """Processes real and generated samples through multiple discriminators."""
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for discriminator in self.discriminators:
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
