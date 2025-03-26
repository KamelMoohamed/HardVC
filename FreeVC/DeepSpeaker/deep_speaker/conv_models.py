import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from .constants import NUM_FBANKS, SAMPLE_RATE


class Normalize(nn.Module):
    def __init__(self, dim=1, eps=1e-12, adjusted=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.adjusted = adjusted

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        std = x.std(dim=self.dim, keepdim=True)
        if self.adjusted:
            min_std = 1.0 / math.sqrt(x[0].numel())
            std = torch.clamp(std, min=min_std)
        if self.eps > 0:
            std = torch.clamp(std, min=self.eps)
        return (x - mean) / std


class ClippedReLU(nn.Module):
    def __init__(self, max_value=20):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.max_value)


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, filters, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = ClippedReLU()

        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x += residual
        return self.relu(x)


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=2, padding=2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = ClippedReLU()
        self.res_blocks = nn.Sequential(
            IdentityBlock(out_channels, out_channels, 3),
            IdentityBlock(out_channels, out_channels, 3),
            IdentityBlock(out_channels, out_channels, 3),
        )

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.res_blocks(x)


class DeepSpeakerModel(nn.Module):
    def __init__(self, include_softmax=False, num_speakers_softmax=None):
        super().__init__()
        self.include_softmax = include_softmax
        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            hop_length=int(0.01 * SAMPLE_RATE),
            win_length=int(0.025 * SAMPLE_RATE),
            n_mels=NUM_FBANKS,
        )

        self.normalize = Normalize(dim=-1)
        self.conv_blocks = nn.Sequential(
            ConvResBlock(1, 64),
            ConvResBlock(64, 128),
            ConvResBlock(128, 256),
            ConvResBlock(256, 512),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(2048, 512)

        if self.include_softmax:
            assert num_speakers_softmax is not None
            self.dropout = nn.Dropout(0.5)
            self.output = nn.Linear(512, num_speakers_softmax)
        else:
            self.output = nn.Identity()

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.mel(x)
            x = self.normalize(x)
            x = x.unsqueeze(1)

        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1, 2048)
        x = x.mean(dim=1)
        if self.include_softmax:
            x = self.dropout(x)
        x = self.dense1(x)
        if self.include_softmax:
            x = self.output(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x
