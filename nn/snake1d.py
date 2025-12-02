import torch
import torch.nn as nn


def snake1d(
        x: torch.Tensor,
        alpha: torch.Tensor,
        eps: float = 1e-9
) -> torch.Tensor:
    if x.ndim != 3:
        # Expected (B, C, L) format
        raise ValueError

    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + eps).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-9) -> None:
        super().__init__()

        # Params
        self.num_channels = num_channels
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake1d(x, alpha=self.alpha, eps=self.eps)
    