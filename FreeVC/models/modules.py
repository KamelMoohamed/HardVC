import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm

from commons import fused_add_tanh_sigmoid_multiply, get_padding, init_weights

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.transpose(1, -1), (x.shape[-1],), self.gamma, self.beta, self.eps
        ).transpose(1, -1)


class ConvReluNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
    ):
        super().__init__()
        assert n_layers > 1, "Number of layers should be larger than 1."

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
                )
            ]
        )
        self.norm_layers = nn.ModuleList([LayerNorm(hidden_channels)])
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))

        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x_org = x
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = conv(x * x_mask)
            x = norm(x)
            x = self.relu_drop(x)
        return (x_org + self.proj(x)) * x_mask


class DDSConv(nn.Module):
    """
    Dilated and Depth-Separable Convolution
    """

    def __init__(
        self, channels: int, kernel_size: int, n_layers: int, p_dropout: float = 0.0
    ):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()

        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2

            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor = None
    ) -> torch.Tensor:
        if g is not None:
            x = x + g

        for conv_sep, conv_1x1, norm1, norm2 in zip(
            self.convs_sep, self.convs_1x1, self.norms_1, self.norms_2
        ):
            y = conv_sep(x * x_mask)
            y = norm1(y)
            y = F.gelu(y)
            y = conv_1x1(y)
            y = norm2(y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y

        return x * x_mask


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."

        self.hidden_channels = hidden_channels
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.drop = nn.Dropout(p_dropout)

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        if gin_channels:
            self.cond_layer = nn.utils.weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1),
                name="weight",
            )

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor = None
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
        if g is not None:
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
        else:
            g_l = torch.zeros_like(x_in)

        acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
        acts = self.drop(acts)

        res_skip_acts = self.res_skip_layers[i](acts)
        if i < self.n_layers - 1:
            res_acts = res_skip_acts[:, : self.hidden_channels, :]
            x = (x + res_acts) * x_mask
            output = output + res_skip_acts[:, self.hidden_channels :, :]
        else:
            output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels:
            nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            nn.utils.remove_weight_norm(layer)


class ResBlock1(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=(kernel_size - 1) * d // 2,
                    )
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=(kernel_size - 1) // 2,
                    )
                )
                for _ in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt *= x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            if x_mask is not None:
                xt *= x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x *= x_mask
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt *= x_mask
            x = xt + conv(xt)
        if x_mask is not None:
            x *= x_mask
        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            remove_weight_norm(conv)


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if reverse:
            return torch.exp(x) * x_mask
        y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
        logdet = -torch.sum(y, dim=[1, 2])
        return y, logdet


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if reverse:
            return x
        logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        return x, logdet


class ElementWiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if reverse:
            return (x - self.m) * torch.exp(-self.logs) * x_mask
        y = (self.m + torch.exp(self.logs) * x) * x_mask
        logdet = torch.sum(self.logs * x_mask, dim=[1, 2])
        return y, logdet


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "Channels should be divisible by 2"
        super().__init__()

        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, kernel_size=1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), kernel_size=1
        )

        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, self.half_channels, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if self.mean_only:
            m = stats
            logs = torch.zeros_like(m)
        else:
            m, logs = torch.split(stats, self.half_channels, dim=1)

        if reverse:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
        else:
            x1 = m + x1 * torch.exp(logs) * x_mask
            logdet = torch.sum(logs, dim=[1, 2])
            return torch.cat([x0, x1], dim=1), logdet

        return torch.cat([x0, x1], dim=1)
