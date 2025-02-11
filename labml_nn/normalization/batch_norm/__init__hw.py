import torch
from torch import nn

from labml_helpers.module import Module

class BatchNorm(nn.Module):
    r"""
    Homework: Implementing and Extending Batch Normalization

    This class implements Batch Normalization that can handle different input shapes.

    **Supported Input Shapes**:
    1) (B, C, H, W) for image data.
    2) (B, C) for embeddings.
    3) (B, C, L) for sequence data.

    Where:
    - B = batch size,
    - C = number of channels or features,
    - H, W = spatial dimensions (for images),
    - L = sequence length (for sequential data).

    Formula:
    BN(X) = gamma * (X - E[X]) / sqrt(Var[X] + eps) + beta

    The trainable parameters (if affine=True) are:
    - scale (gamma), shape = (C,)
    - shift (beta), shape = (C,)

    Also tracks running estimates of mean and variance if track_running_stats=True.

    TODO:
    1) Add additional comments on how to handle corner cases (like B=1).
    2) Investigate the effect of the momentum hyperparameter on your training.
    3) Provide options for channel-last data (B, H, W, C) if desired.
    4) Consider numeric stability improvements (e.g., alternative ways to calculate variance).
    """

    def __init__(
        self,
        channels: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        """
        Constructor for BatchNorm.

        Args:
            channels (int): Number of features in the input, matching the size of the second dimension.
            eps (float): Small constant added to variance to avoid division by zero. Default: 1e-5.
            momentum (float): Factor for updating running averages. Default: 0.1.
            affine (bool): If True, this module has learnable scale (gamma) and shift (beta). Default: True.
            track_running_stats (bool): If True, track moving averages of mean/variance. Default: True.

        Input Shape:
            - (B, C, *) where B is batch size, C is `channels`, * can be zero or more extra dims.

        Output:
            None (initializes parameters/buffers for BN).

        TODO:
            1) Validate that `channels` is positive and matches the input shape in forward.
            2) Optionally initialize scale and shift differently (e.g., scale=0.5, shift=0.0).
            3) Possibly allow custom initialization strategies via extra arguments.
        """
        super().__init__()

        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BatchNorm.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, *), where:
                            - B is the batch size,
                            - C = self.channels,
                            - * = any number of additional dimensions (e.g. H, W or L).
                            Type: float tensor.

        Returns:
            torch.Tensor: Batch-normalized output with the same shape (B, C, *), type float tensor.

        TODO:
            1) Distinguish behavior for training vs. inference modes in more detail.
            2) Investigate numerical issues when batch_size is very small.
            3) Optionally add a mechanism to override the default PyTorch .training flag.
            4) Provide hooks or logs for debugging the computed mean and variance.
        """
        pass
