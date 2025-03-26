from torch import nn

from .modules import Flip, ResidualCouplingLayer


class ResidualCouplingBlock(nn.Module):
    """
    Residual Coupling Block for Flow-based models.

    This block consists of multiple residual coupling layers interleaved with Flip operations.

    Args:
        channels (int): Number of input/output channels.
        hidden_channels (int): Number of hidden channels in the coupling layers.
        kernel_size (int): Kernel size for convolutional layers.
        dilation_rate (int): Dilation rate for convolutions.
        n_layers (int): Number of layers in each residual coupling layer.
        n_flows (int, optional): Number of coupling layers. Defaults to 4.
        gin_channels (int, optional): Number of channels for global conditioning. Defaults to 0.
    """

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()

        # Create a sequence of coupling layers followed by Flip layers
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
        self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Forward pass for residual coupling transformation.

        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, time_steps).
            x_mask (torch.Tensor): Mask tensor for valid elements.
            g (torch.Tensor, optional): Global conditioning tensor. Defaults to None.
            reverse (bool, optional): If True, runs the flow in reverse. Defaults to False.

        Returns:
            torch.Tensor: Transformed output tensor.
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
