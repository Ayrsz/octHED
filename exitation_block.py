import torch
import torch.nn as nn

class SqueezeExitationBlock(nn.Module):
    def __init__(self, in_channels: int):
        """Constructor for SqueezeExitationBlock.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(
            in_channels, in_channels // 4
        )  # divide by 4 is mentioned in the paper, 5.3. Large squeeze-and-excite
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(in_channels // 4, in_channels)
        self.act2 = nn.Hardsigmoid()

    def forward(self, x):
        """Forward pass for SqueezeExitationBlock."""

        identity = x

        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)

        x = identity * x[:, :, None, None]

        return x