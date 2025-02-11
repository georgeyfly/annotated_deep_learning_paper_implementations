import torch
from torch import nn

from labml_helpers.module import Module

class BatchNorm(Module):
    """
    Batch Normalization Layer Implementation
        
    This layer normalizes input features across the batch dimension to reduce 
    internal covariate shift during training. It supports multiple input formats:
        
    1. Image input: (B, C, H, W) - batch of image representations
    2. Linear input: (B, C) - batch of embeddings
    3. Sequence input: (B, C, L) - batch of sequence embeddings
        
    where:
        B = batch size
        C = number of channels/features
        H = height (for images)
        W = width (for images)
        L = sequence length
        
    Mathematical formulation:
    BN(x) = γ * ((x - E[x]) / sqrt(Var[x] + ε)) + β
        
    where:
        γ (scale) and β (shift) are learnable parameters
        E[x] is the mean computed over the batch dimension
        Var[x] is the variance computed over the batch dimension
        ε is a small constant for numerical stability
    """

    def __init__(self, channels: int, *,
                eps: float = 1e-5, momentum: float = 0.1,
                affine: bool = True, track_running_stats: bool = True):
        """
        Initialize the BatchNorm layer.
            
        Args:
            channels (int): Number of features/channels in the input
            eps (float): Small constant for numerical stability
            momentum (float): Momentum factor for running statistics
            affine (bool): If True, use learnable affine parameters
            track_running_stats (bool): If True, maintain running statistics
                
        Returns:
            None
                
        TODO:
        1. Initialize parent class
        2. Store all input parameters as instance variables
        3. If affine is True:
        - Create learnable parameters 'scale' (γ) initialized to ones
        - Create learnable parameters 'shift' (β) initialized to zeros
        4. If track_running_stats is True:
        - Create buffer for exponential moving average of mean
        - Create buffer for exponential moving average of variance
        """
        # Initialize parent class
        super().__init__()

        # Store parameters
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # TODO: Initialize learnable parameters and buffers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform batch normalization on the input tensor.
            
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, *]
                            where * represents any number of additional dimensions
            
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input
                
        TODO:
        1. Input validation:
        - Store original input shape
        - Verify channel dimension matches self.channels
        - Reshape input to [batch_size, channels, -1]
            
        2. Statistics computation:
        If training or not tracking running stats:
        - Compute batch mean across [batch, -1] dimensions
        - Compute batch variance across [batch, -1] dimensions
        - If tracking stats, update exponential moving averages
        Else:
        - Use stored running statistics
            
        3. Normalization:
        - Normalize input using computed/stored statistics
        - Apply epsilon for numerical stability
            
        4. Scale and shift:
        If affine is True:
        - Apply learnable scale and shift parameters
            
        5. Reshape output:
        - Restore original input shape
        - Return normalized tensor
        """
        # Store original shape and validate input
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]

        # TODO: Implement batch normalization logic

        return None  # Replace with actual normalized tensor


def _test():
    """
    Simple test
    """
    from labml.logger import inspect

    x = torch.zeros([2, 3, 2, 4])
    inspect(x.shape)
    bn = BatchNorm(3)

    x = bn(x)
    inspect(x.shape)
    inspect(bn.exp_var.shape)


#
if __name__ == "__main__":
    _test()
